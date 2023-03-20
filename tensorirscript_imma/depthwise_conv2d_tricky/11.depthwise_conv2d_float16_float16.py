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
import sys
# add path ../..
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tvm.tir.tensor_intrin.cuda import (
    WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
    WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
    WMMA_FILL_16x16x16_F16_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
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


VERIFY = True

# The sizes of inputs and filters
batch_size = 128
height = 224
width = 224
in_channels = 84
kernel_h = 3
kernel_w = 3
pad_h = 0
pad_w = 0
stride_h = 1
stride_w = 1
dilation_h = 1
dilation_w = 1
factor = 1
assert factor == 1, "we currently only support factor == 1"
out_dtype = "float16"
out_channels = in_channels * factor

output_height = (height + 2 * pad_h - dilation_h *
                 (kernel_h - 1) - 1) // stride_h + 1
output_width = (width + 2 * pad_w - dilation_w *
                (kernel_w - 1) - 1) // stride_w + 1
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
raster = 8
stage = 1

vec = 8
wmma_m = 16
wmma_n = 16
wmma_k = 16


# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size,
    height,
    width,
    in_channels
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    in_channels,
    kernel_h,
    kernel_w,
    factor,
)
# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size,
    output_height,
    output_width,
    out_channels,
)
BATCH = in_channels
M = batch_size * output_height * output_width
N = factor
K = kernel_h * kernel_w

# Algorithm
A = te.placeholder(data_shape, name="A", dtype="float16")
W = te.placeholder(kernel_shape, name="W", dtype="float16")
kh = te.reduce_axis((0, kernel_h), name="kh")
kw = te.reduce_axis((0, kernel_w), name="kw")
Apad = te.compute(
    (
        batch_size,
        height + 2 * pad_h,
        width + 2 * pad_w,
        in_channels,
    ),
    lambda n, h, w, i: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height,
                    w >= pad_w, w - pad_w < width),
        A[n, h - pad_h, w - pad_w, i],
        tvm.tir.const(0.0, "float16"),
    ),
    name="Apad",
)
A_Imp_shape = (BATCH, M, K)
A_Imp = te.compute(
    A_Imp_shape,
    lambda b, x, y: Apad[
        x // (output_height * output_width),
        stride_h * ((x % (output_height * output_width)) // output_width) +
        dilation_h * (y // kernel_w),
        stride_w * ((x % (output_height * output_width)) %
                    output_width) + dilation_w * (y % kernel_w),
        b,
    ],
    name="data_im2col",
)
W_flat_shape = (N, K)
W_flat = te.compute(
    W_flat_shape, lambda n, k: W[n, (k // in_channels) // kernel_w,
                                 (k // in_channels) % kernel_w, k % in_channels], "weight_flatten"
)

k = te.reduce_axis((0, K), name="k")
Conv = te.compute(
    (BATCH, M, N),
    lambda b, x, y: te.sum(A_Imp[b, x, k].astype(out_dtype)
                        * W_flat[y, k].astype(out_dtype), axis=k),
    name="depth_conv2d_nhwc",
)

ir_module = te.create_prim_func([A, W, Conv])
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("Apad")
block_im2col = sch.get_block("data_im2col")
block_flat = sch.get_block("weight_flatten")
block_conv = sch.get_block("depth_conv2d_nhwc")
write_sch(sch, log_path, "cache_related")

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")
sch.compute_inline(block_im2col)
write_sch(sch, log_path, "Im2ColInline")
sch.compute_inline(block_flat)
write_sch(sch, log_path, "FlatInline")

(b, i, j, k) = sch.get_loops(block_conv)

# bx, tx = sch.split(i, factors=[None, 1024])
sch.bind(i, "blockIdx.x")
sch.bind(b, "threadIdx.x")
write_sch(sch, log_path, "tricky_extract_compute")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# a_np = np.ones(data_shape).astype("float16")
b_np = np.ones(kernel_shape).astype("float16")
# random init a and b
a_np = np.random.uniform(size=data_shape).astype("float16") * 127
# b_np = np.random.uniform(size=kernel_shape).astype("float16") * 127
# aragnge init a and b
# a_np = np.mod(np.arange(0, batch_size * in_channels * height * width), 4).reshape(data_shape).astype("float16")
# b_np = np.mod(np.arange(0, kernel_h * kernel_w * in_channels * factor), 4).reshape(kernel_shape).astype("float16")

c_np = np.zeros((BATCH, M, N)).astype(out_dtype)
cuda_a = tvm.nd.array(a_np, ctx)
cuda_b = tvm.nd.array(b_np, ctx)
cuda_c = tvm.nd.array(c_np, ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)


if VERIFY:
    # do conv with torch
    import torch
    # convert a from nhwc to nchw
    a_np = np.transpose(a_np, (0, 3, 1, 2))
    a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float32)
    a_torch = torch.nn.functional.pad(a_torch, (pad_h, pad_h, pad_w, pad_w))
    # convert b from ohwi to oihw
    b_np = np.transpose(b_np, (0, 3, 1, 2))
    b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float32)
    m = torch.nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(pad_h, pad_w),
                        groups=in_channels, dtype=torch.float32).cuda()
    m.weight.data = b_torch.data
    m.bias.data = torch.zeros(out_channels, dtype=torch.float32).cuda()
    c_torch = m(a_torch)
    print(c_torch.shape)
    c_torch_np = np.transpose(c_torch.detach().cpu().numpy(), (0, 2, 3, 1))
    # nhwc to cnhw
    c_torch_np = np.transpose(c_torch_np, (3, 0, 1, 2))
    c_torch_np = c_torch_np.reshape((BATCH * M *N))
    print("torch result: ", c_torch_np[0:10])
    print("tvm result: ", cuda_c.asnumpy().reshape((BATCH * M * N))[0:10])
    # print("verify result: ", np.allclose(c_torch_np, cuda_c.asnumpy()))

# num_runs = 3
# timer_cuda_mod = cuda_mod.time_evaluator(
#     cuda_mod.entry_name, ctx, number=num_runs)

# t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

# print("convert conv2d into B %d M %d N %d K %d 's gemm, average time cost of %d runs = %g ms" %
    #   (batch_size, M, N, K, num_runs, t * 1e3))
