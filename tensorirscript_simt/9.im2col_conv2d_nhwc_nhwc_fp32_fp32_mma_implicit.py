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


log_path = "progress/tensorir_script/9.im2col_nhwc_nhwc_conv2d_fp32_fp32_mma_implicit"
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

VERIFY = True

# The sizes of inputs and filters
batch_size = 128
height = 42
width = 42
in_channels = 1024
out_channels = 384
kernel_h = 1
kernel_w = 1
pad_h = 0
pad_w = 0
stride_h = 1
stride_w = 1
dilation_h = 1
dilation_w = 1
output_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
output_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
print("output_height: ", output_height)
print("output_width: ", output_width)
# TensorCore shape
wmma_m = 16
wmma_n = 16
wmma_k = 16

# tuning params
warp_size = 32
block_row_warps = 4
block_col_warps = 2
warp_row_tiles = 4
warp_col_tiles = 4
chunk = 2
raster = 1
stage = 1
vec = 4

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size,
    height,
    width,
    in_channels
)

# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    out_channels,
    kernel_h,
    kernel_w,
    in_channels,
)

# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size,
    output_height,
    output_width,
    out_channels,
)
M = batch_size * output_height * output_width
N = out_channels
K = kernel_h * kernel_w * in_channels
print("M: ", M)
print("N: ", N)
print("K: ", K)
# Reduction axes
k = te.reduce_axis((0, K), name="k")

# Algorithm
A = te.placeholder(data_shape, name="A", dtype="float32")
W = te.placeholder(kernel_shape, name="W", dtype="float32")
Apad = te.compute(
    (
        batch_size,
        height + 2 * pad_h,
        width + 2 * pad_w,
        in_channels,
    ),
    lambda n, h, w, i: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height, w >= pad_w, w - pad_w < width),
        A[n, h - pad_h, w - pad_w, i],
        tvm.tir.const(0.0, "float32"),
    ),
    name="Apad",
)
A_Imp_shape = (M, K)
A_Imp = te.compute(
            A_Imp_shape,
            lambda x, y: Apad[
                x // (output_height * output_width),
                stride_h * ((x % (output_height * output_width)) // output_width) + dilation_h * ((y // in_channels) // kernel_w),
                stride_w * ((x % (output_height * output_width)) % output_width) + dilation_w * ((y // in_channels) % kernel_w),
                y % in_channels,
            ],
            name="data_im2col",
        )
W_flat_shape = (N, K)
W_flat = te.compute(
        W_flat_shape, lambda n, k: W[n, (k // in_channels) // kernel_w, (k // in_channels) % kernel_w, k % in_channels], "weight_flatten"
    )
Conv = te.compute(
    (M, N),
    lambda x, y: te.sum(A_Imp[x, k] * W_flat[y, k], axis=k),
    name="Conv",
)

ir_module = te.create_prim_func([A, W, Conv])

print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("Apad")
block_im2col = sch.get_block("data_im2col")
block_flat = sch.get_block("weight_flatten")
block_conv = sch.get_block("Conv")
block_conv_input_shared = sch.cache_read(block_conv, 0 ,"shared")
block_conv_input_frag = sch.cache_read(block_conv, 0, "local")
block_conv_weight_shared = sch.cache_read(block_conv, 1 ,"shared")
block_conv_weight_frag = sch.cache_read(block_conv, 1, "local")
block_conv_output_frag = sch.cache_write(block_conv, 0, "local")
write_sch(sch, log_path, "cache_related")

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")
sch.compute_inline(block_im2col)
write_sch(sch, log_path, "Im2ColInline")
sch.compute_inline(block_flat)
write_sch(sch, log_path, "FlatInline")


(i, j, k) = sch.get_loops(block_conv)
write_sch(sch, log_path, "tricky_extract_compute")

block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj)

write_sch(sch, log_path, "block_tile")

sch.bind(block_i, "blockIdx.y")
sch.bind(block_j, "blockIdx.x")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.x")

write_sch(sch, log_path, "thread_bind")

# cache read A from global memory to shared_memory
sch.compute_at(block_conv_input_frag, ki, preserve_unit_loops=True)
sch.compute_at(block_conv_input_shared, ko, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_frag, ki, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_shared, ko, preserve_unit_loops=True)
sch.reverse_compute_at(block_conv_output_frag, j, preserve_unit_loops=True)
write_sch(sch, log_path, "cache_read_compute_at")

# 128x32
A_shared_fused = sch.fuse(*sch.get_loops(block_conv_input_shared)[-2:])
A_shared_ty, A_shared_tx, A_shared_inner, A_shared_vi = sch.split(
    A_shared_fused, factors=[block_row_warps, block_col_warps, None, vec])
sch.vectorize(A_shared_vi)
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
write_sch(sch, log_path, "schedule_A_shared")

B_shared_fused = sch.fuse(*sch.get_loops(block_conv_weight_shared)[-2:])
B_shared_ty, B_shared_tx, B_shared_inner, B_shared_vi = sch.split(
    B_shared_fused, factors=[block_row_warps, block_col_warps, None, vec])
sch.vectorize(B_shared_vi)
sch.bind(B_shared_tx, "threadIdx.x")
sch.bind(B_shared_ty, "threadIdx.y")

write_sch(sch, log_path, "schedule_B_shared")

# decompose reduction
init_block_b = sch.decompose_reduction(block_conv, ko)
write_sch(sch, log_path, "decompose_reduction")


init_block_b_loops = sch.get_loops(init_block_b)
init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]



# unroll
# write_sch(sch, log_path, "do_unroll")

# if stage > 1:
#     sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage-1])
#     sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
# if raster > 0:
#     sch.annotate(init_block_b_i, ann_key="thread_rasterization", ann_val=raster)


ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# a_np = np.ones(data_shape).astype("float32")
# b_np = np.ones(kernel_shape).astype("float32")
# random init a and b
# a_np = np.random.uniform(size=data_shape).astype("float32")
# b_np = np.random.uniform(size=kernel_shape).astype("float32")
# aragnge init a and b
a_np = np.mod(np.arange(0, batch_size * in_channels * height * width), 10).reshape(data_shape).astype("float32")
b_np = np.mod(np.arange(0, kernel_h * kernel_w * in_channels * out_channels), 10).reshape(kernel_shape).astype("float32")

c_np = np.zeros((M, N)).astype("float32")
cuda_a = tvm.nd.array(a_np, ctx)
cuda_b = tvm.nd.array(b_np, ctx)
cuda_c = tvm.nd.array(c_np, ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)


if VERIFY:
    # do conv with torch
    import torch
    # convert a from nhwc to nchw
    a_np = np.transpose(a_np, (0, 3, 1, 2))
    a_torch = torch.tensor(a_np, device="cuda")
    a_torch = torch.nn.functional.pad(a_torch, (pad_h, pad_h, pad_w, pad_w))
    # convert b from ohwi to oihw
    b_np = np.transpose(b_np, (0, 3, 1, 2))
    b_torch = torch.tensor(b_np, device="cuda")
    c_torch = torch.nn.functional.conv2d(a_torch, b_torch, groups=1)
    c_torch_np = np.transpose(c_torch.cpu().numpy(), (0, 2, 3, 1))
    c_torch_np = c_torch_np.reshape((M, N))    
    print("torch result: ", c_torch_np[0][0:10])
    print("tvm result: ", cuda_c.asnumpy()[0][0:10])
    np.testing.assert_allclose(c_torch_np, cuda_c.asnumpy(), rtol=1e-5)
    print("verify result: ", np.allclose(c_torch_np, cuda_c.asnumpy()))

num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

print("convert conv2d into B %d M %d N %d K %d 's gemm, average time cost of %d runs = %g ms" %
      (batch_size, M, N, K, num_runs, t * 1e3))
