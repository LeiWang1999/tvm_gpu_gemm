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
log_path = "progress/tensorirscript_simt/" + fname

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
height = 83
width = 83
in_channels = 84
kernel_h = 5
kernel_w = 5
pad_h = 2
pad_w = 2
stride_h = 2
stride_w = 2
dilation_h = 1
dilation_w = 1
factor = 1
out_dtype = "int32"
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

vec = 16
wmma_m = 16
wmma_n = 16
wmma_k = 16


# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    in_channels,
    batch_size,
    height,
    width,
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
    out_channels,
    batch_size,
    output_height,
    output_width,
)
M = batch_size * output_height * output_width
N = out_channels
K = kernel_h * kernel_w * in_channels


# Algorithm
A = te.placeholder(data_shape, name="A", dtype="int8")
W = te.placeholder(kernel_shape, name="W", dtype="int8")
kh = te.reduce_axis((0, kernel_h), name="kh")
kw = te.reduce_axis((0, kernel_w), name="kw")
Apad = te.compute(
    (
        in_channels,
        batch_size,
        height + 2 * pad_h,
        width + 2 * pad_w,
    ),
    lambda n, h, w, i: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height,
                    w >= pad_w, w - pad_w < width),
        A[i, n, h - pad_h, w - pad_w],
        tvm.tir.const(0.0, "int8"),
    ),
    name="Apad",
)
Conv = te.compute(
    (batch_size, output_height, output_width, in_channels, factor),
    lambda co, n, h, w, ci: te.sum(
        tir.Cast(
            out_dtype,
            Apad[co, n, h * stride_h + kh * dilation_h,
                 w * stride_w + kw * dilation_w],
        )
        * tir.Cast(out_dtype, W[co, kh, kw, ci]),
        axis=[kh, kw],
    ),
    name="depth_conv2d_nhwc",
)
# reshape_output = te.compute(
#     (batch_size, output_height, output_width, out_channels),
#     lambda c, n, h, w: Conv[te.indexdiv(
#         c, factor), n, h, w, te.indexmod(c, factor)],
#     name="reshape_output",
# )

ir_module = te.create_prim_func([A, W, Conv])
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("Apad")
block_conv = sch.get_block("depth_conv2d_nhwc")
# block_reshape = sch.get_block("reshape_output")
block_conv_input_shared = sch.cache_read(block_conv, 0, "shared")
# block_conv_input_frag = sch.cache_read(block_conv, 0, "local")
block_conv_weight_shared = sch.cache_read(block_conv, 1, "shared")
# block_conv_weight_frag = sch.cache_read(block_conv, 1, "local")
block_conv_output_frag = sch.cache_write(block_conv, 0, "local")
write_sch(sch, log_path, "cache_related")

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")
# sch.reverse_compute_inline(block_reshape)
write_sch(sch, log_path, "ReshapeInline")

# unroll
ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# a_np = np.ones(data_shape).astype("int8")
# b_np = np.ones(kernel_shape).astype("int8")
# random init a and b
a_np = np.random.uniform(size=data_shape).astype("int8")
b_np = np.random.uniform(size=kernel_shape).astype("int8")
# aragnge init a and b
# a_np = np.mod(np.arange(0, batch_size * in_channels * height * width), 10).reshape(data_shape).astype("int8")
# b_np = np.mod(np.arange(0, kernel_h * kernel_w * in_channels * out_channels), 10).reshape(kernel_shape).astype("int8")

c_np = np.zeros((M, N)).astype("int8")
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
    print("verify result: ", np.allclose(c_torch_np, cuda_c.asnumpy()))

num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

print("convert conv2d into B %d M %d N %d K %d 's gemm, average time cost of %d runs = %g ms" %
      (batch_size, M, N, K, num_runs, t * 1e3))
