"""
    Considering a gemm problem, in this part we try to leverage the ldmatrix, mma, and stmatrix to do the computation.
    The ldmatrix and stmatrix are used to load and store the data from global memory to shared memory.
    The mma is used to do the computation.
    thread_x will be set into 32, which represents the number of threads in a warp.
    thread_y and thread_z will be set into value which represents the array of warps. 
"""
import tvm
from tvm.script import tir as T
from tvm import te, tir, topi
import numpy as np
import os
from tvm.tir.tensor_intrin.cuda import (
    WMMA_FILL_16x16x16_S32_INTRIN,
    WMMA_LOAD_16x16x16_S8_A_INTRIN,
    WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN,
    WMMA_LOAD_16x16x16_S8_B_INTRIN,
    WMMA_SYNC_16x16x16_s8s8s32_INTRIN,
    WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN,
    WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN
)


# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/tensorirscript_imma/" + fname
count = 0
def write_code(code, path, fname):
    global count
    # if path not exist, create it
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)

def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)


VERIFY = False

# The sizes of inputs and filters
batch_size = 128
height = 56
width = 56
in_channels = 256
out_channels = 512
kernel_h = 1
kernel_w = 1
pad_h = 0
pad_w = 0
stride_h = 2
stride_w = 2

# TensorCore shape
wmma_m = 16
wmma_n = 16
wmma_k = 16


assert batch_size % wmma_k == 0
assert in_channels % wmma_m == 0
assert out_channels % wmma_n == 0

# tuning params
block_row_warps = 4
block_col_warps = 1
warp_row_tiles = 1
warp_col_tiles = 16
warp_size = 32
chunk = 2
stage = 1
raster = 8

vec = 16
output_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
output_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
print("output_height: ", output_height)
print("output_width: ", output_width)

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    height,
    width,
    batch_size // wmma_m,
    in_channels // wmma_k,
    wmma_m,
    wmma_k,
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    kernel_h,
    kernel_w,
    out_channels // wmma_n,
    in_channels // wmma_k,
    wmma_n,
    wmma_k,
)
# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    output_height,
    output_width,
    batch_size // wmma_m,
    out_channels // wmma_n,
    wmma_m,
    wmma_n,
)
# Reduction axes
kh = te.reduce_axis((0, kernel_h), name="kh")
kw = te.reduce_axis((0, kernel_w), name="kw")
ic = te.reduce_axis((0, in_channels // wmma_k), name="ic")
ii = te.reduce_axis((0, wmma_k), name="ii")
# Algorithm
A = te.placeholder(data_shape, name="A", dtype="int8")
W = te.placeholder(kernel_shape, name="W", dtype="int8")
APad = te.compute(
    (height + 2 * pad_h, width + 2 * pad_w, batch_size // wmma_m, in_channels // wmma_k, wmma_m, wmma_k),
    lambda h, w, n, ic, nn, ii: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h < height + pad_h, w >= pad_w, w < width + pad_w),
        A[h - pad_h, w - pad_w, n, ic, nn, ii],
        tvm.tir.const(0, "int8"),
    ),
    name="APad",
)
Conv = te.compute(
    output_shape,
    lambda h, w, n, o, nn, oo: te.sum(
        APad[h * stride_h + kh, w * stride_w + kw, n, ic, nn, ii].astype("int32")
        * W[kh, kw, o, ic, oo, ii].astype("int32"),
        axis=[ic, kh, kw, ii],
    ),
    name="Conv",
)

ir_module = te.create_prim_func([A, W, Conv])

print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("APad")
block_conv = sch.get_block("Conv")
block_conv_input_shared = sch.cache_read(block_conv, 0 ,"shared")
block_conv_input_frag = sch.cache_read(block_conv, 0, "wmma.matrix_a")
block_conv_weight_shared = sch.cache_read(block_conv, 1 ,"shared")
block_conv_weight_frag = sch.cache_read(block_conv, 1, "wmma.matrix_b")
block_conv_output_frag = sch.cache_write(block_conv, 0, "wmma.accumulator")
write_sch(sch, log_path, "cache_related")

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")

hc, wc, nc, oc, nnc, ooc, ic, kh, kw, ii = sch.get_loops(block_conv)
block_k = sch.fuse(hc, wc)
sch.bind(block_k, "blockIdx.y")
nc, nci = sch.split(nc, factors=[None, warp_row_tiles])
block_i, nc = sch.split(nc, factors=[None, block_row_warps])
oc, oci = sch.split(oc, factors=[None, warp_col_tiles])
block_j, oc = sch.split(oc, factors=[None, block_col_warps])
sch.reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
block_i_j = sch.fuse(block_i, block_j)
sch.bind(block_i_j, "blockIdx.y")
sch.bind(nc,"threadIdx.y")
sch.bind(oc,"threadIdx.z")
write_sch(sch, log_path, "schedule_block_conv")

sch.reverse_compute_at(block_conv_output_frag, oc, preserve_unit_loops=True)
n_1, o_1, nn, oo, ic, kh, kw, ii = sch.get_loops(block_conv)[-8:]
ko, ki = sch.split(ic, factors=[None, chunk])
sch.reorder(ko, kh, ki, kw, n_1, o_1, nn, oo, ii)
ko = sch.fuse(ko, kh)
write_sch(sch, log_path, "reverse_compute_at_output_frag")

sch.compute_at(block_conv_input_frag, kw, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_frag, kw, preserve_unit_loops=True)
write_sch(sch, log_path, "compute_at_input_frag")

def schedule_shared_A(block):
    sch.compute_at(block, ko, preserve_unit_loops=True)
    A_shared_i, A_shared_j = sch.get_loops(block_conv_input_shared)[-2:]
    A_shared_i_j = sch.fuse(A_shared_i, A_shared_j)
    A_shared_i_j, A_shared_vi = sch.split(A_shared_i_j, factors=[None, vec])
    sch.vectorize(A_shared_vi)
    A_shared_fused = sch.fuse(
        *sch.get_loops(block_conv_input_shared)[-5:-2], A_shared_i_j)
    A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx = sch.split(
        A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size])
    sch.bind(A_shared_tx, "threadIdx.x")
    sch.bind(A_shared_ty, "threadIdx.y")
    sch.bind(A_shared_tz, "threadIdx.z")


def schedule_shared_B(block):
    sch.compute_at(block, ko, preserve_unit_loops=True)
    B_shared_fused = sch.fuse(*sch.get_loops(block_conv_weight_shared)[-5:])
    B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
        B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
    sch.vectorize(B_shared_vi)
    sch.bind(B_shared_tx, "threadIdx.x")
    sch.bind(B_shared_ty, "threadIdx.y")
    sch.bind(B_shared_tz, "threadIdx.z")

# schedule the shared memory into A
schedule_shared_A(block_conv_input_shared)
write_sch(sch, log_path, "schedule_A_shared")

# schedule the shared memory into B
schedule_shared_B(block_conv_weight_shared)
write_sch(sch, log_path, "schedule_B_shared")

init_block_conv = sch.decompose_reduction(block_conv, ko)
init_block_i, init_block_j = sch.get_loops(init_block_conv)[-4:-2]
write_sch(sch, log_path, "decompose_reduction")

sch.tensorize(sch.get_loops(init_block_conv)[-2], WMMA_FILL_16x16x16_S32_INTRIN)
write_sch(sch, log_path, "tensorize_wmma_fill")


sch.tensorize(sch.get_loops(block_conv_input_frag)
              [-2], WMMA_LOAD_16x16x16_S8_A_INTRIN)
sch.tensorize(sch.get_loops(block_conv_weight_frag)
              [-2], WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN)
write_sch(sch, log_path, "tensorize_ldmatrix")

sch.tensorize(nn, WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN)
write_sch(sch, log_path, "tensorize_wmma_sync")

sch.tensorize(sch.get_loops(block_conv_output_frag)[-2], WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN)
write_sch(sch, log_path, "tensorize_store")

# unroll
write_sch(sch, log_path,
          "do_unroll")
if stage > 1:
    sch.annotate(ko, ann_key="software_pipeline_stage",
                 ann_val=[0, 0, stage - 1])
    sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
if raster > 0:
    sch.annotate(init_block_i,
                 ann_key="thread_rasterization", ann_val=raster)

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# a_np = np.ones(data_shape).astype("int8")
# b_np = np.ones(kernel_shape).astype("int8")
# random init a and b
# a_np = np.random.uniform(size=data_shape).astype("int8")
# b_np = np.random.uniform(size=(N // wmma_m, K // wmma_k, wmma_n, wmma_k)).astype("int8")
# aragnge init a and b
a_np = np.mod(np.arange(0, batch_size * in_channels * height *
              width), 4).reshape(data_shape).astype("int8")
b_np = np.mod(np.arange(0, kernel_h * kernel_w * in_channels * out_channels),
  4).reshape(kernel_shape).astype("int8")

c_np = np.zeros(output_shape).astype("int32")
cuda_a = tvm.nd.array(a_np, ctx)
cuda_b = tvm.nd.array(b_np, ctx)
cuda_c = tvm.nd.array(c_np, ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)


if VERIFY:
    # do conv with torch
    import torch
    # convert a from hwnc1616 into nhwc
    a_np = np.transpose(a_np, (2, 4, 0, 1, 3, 5)).reshape(
        (batch_size, height, width, in_channels))
    # convert a from nhwc to nchw
    a_np = np.transpose(a_np, (0, 3, 1, 2))
    a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float32)
    a_torch = torch.nn.functional.pad(a_torch, (pad_h, pad_h, pad_w, pad_w))
    
    # convert b from ohwi1616 into ohwi
    b_np = np.transpose(b_np, (0, 4, 1, 2, 3, 5)).reshape(
        (out_channels, kernel_h, kernel_w, in_channels))
    # convert b from ohwi to oihw
    b_np = np.transpose(b_np, (0, 3, 1, 2))
    b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float32)
    c_torch = torch.nn.functional.conv2d(
        a_torch, b_torch, stride=(stride_h, stride_w), groups=1)
    c_torch_np = np.transpose(c_torch.cpu().numpy(), (0, 2, 3, 1))
    print(c_torch_np.shape)
    # convert c from hwno1616 into nhwc
    c_np = cuda_c.numpy().transpose((2, 4, 0, 1, 3, 5)).reshape(
        (batch_size, output_height, output_width, out_channels)
    )
    print("torch result: ", c_torch_np[0][0][0][0:10])
    print("tvm result: ", c_np[0][0][0][0:10])
    print("verify result: ", np.allclose(c_torch_np, c_np))


num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

print("task conv2d, average time cost of %d runs = %g ms" %
      (num_runs, t * 1e3))
