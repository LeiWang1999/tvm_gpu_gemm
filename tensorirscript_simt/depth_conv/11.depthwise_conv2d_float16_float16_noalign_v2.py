"""
    Considering a gemm problem, in this part we try to leverage the ldmatrix, mma, and stmatrix to do the computation.
    The ldmatrix and stmatrix are used to load and store the data from global memory to shared memory.
    The mma is used to do the computation.
    thread_x will be set into 32, which represents the number of threads in a warp.
    thread_y and thread_z will be set into value which represents the array of warps. 
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tvm
from tvm import te
import numpy as np
import tvm.testing

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


VERIFY = True

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
assert factor == 1, "we currently only support factor == 1"
out_dtype = "float16"
out_channels = in_channels * factor

output_height = (height + 2 * pad_h - dilation_h *
                 (kernel_h - 1) - 1) // stride_h + 1
output_width = (width + 2 * pad_w - dilation_w *
                (kernel_w - 1) - 1) // stride_w + 1
print("output_height: ", output_height)
print("output_width: ", output_width)

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    in_channels,
    batch_size,
    height,
    width
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

M = in_channels * batch_size * output_height * output_width
N = factor
K = kernel_h * kernel_w
# TensorCore shape
wmma_m = 16
wmma_n = 16
wmma_k = 16

# tuning params
num_warps = 4
chunk = 16

warp_size = 32
vec = 16

MMA_M = 1
MMA_N = 1
MMA_K = 4

# pad K to be multiple of MMA_K * 2
# KPAD = K + (MMA_K * 2 - K % (MMA_K * 2)) % (MMA_K * 2)
KPAD = K
print("KPAD: ", KPAD)

if chunk * MMA_K < vec:
    vec = chunk * MMA_K

vec_size = vec // MMA_K
num_tx = K // vec if K // vec < warp_size else warp_size
num_ty = warp_size // num_tx

print("num_tx: ", num_tx)
print("num_ty: ", num_ty)

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
    lambda i, n, h, w: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height,
                    w >= pad_w, w - pad_w < width),
        A[i, n, h - pad_h, w - pad_w],
        tvm.tir.const(0.0, "float16"),
    ),
    name="Apad",
)
A_Imp_shape = (M, K)
A_Imp = te.compute(
    A_Imp_shape,
    lambda x, y: Apad[
        x // (output_height * output_width * batch_size),
        x // (output_height * output_width) % batch_size,
        stride_h * ((x % (output_height * output_width)) // output_width) +
        dilation_h * (y // kernel_w),
        stride_w * ((x % (output_height * output_width)) %
                    output_width) + dilation_w * (y % kernel_w),
    ],
    name="data_im2col",
)
A_Imp_Pad = te.compute(
    (M, KPAD),
    lambda x, y: tvm.tir.if_then_else(
        tvm.tir.all(y < K),
        A_Imp[x, y],
        tvm.tir.const(0.0, "float16"),
    ),
    name="data_im2col_pad",
)
W_flat_shape = (N, K)
W_flat = te.compute(
    W_flat_shape, lambda n, k: W[(k // in_channels) // kernel_w,
                                 (k // in_channels) % kernel_w, k % in_channels, n], "weight_flatten"
)
W_flat_Pad = te.compute(
    (N, KPAD),
    lambda x, y: tvm.tir.if_then_else(
        tvm.tir.all(y < K),
        W_flat[x, y],
        tvm.tir.const(0.0, "float16"),
    ),
    name="weight_flatten_pad",
)


k = te.reduce_axis((0, KPAD), name="k")
Conv = te.compute(
    (M, N),
    lambda x, y: te.sum(A_Imp_Pad[x, k].astype(out_dtype)
                        * W_flat_Pad[y, k].astype(out_dtype), axis=k),
    name="depth_conv2d_nhwc",
)

ir_module = te.create_prim_func([A, W, Conv])
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("Apad")
block_im2col = sch.get_block("data_im2col")
block_im2col_pad = sch.get_block("data_im2col_pad")
block_flat = sch.get_block("weight_flatten")
block_flat_pad = sch.get_block("weight_flatten_pad")
block_conv = sch.get_block("depth_conv2d_nhwc")
block_shared_local_A = sch.cache_read(block_conv, 0, "local")
block_shared_local_B = sch.cache_read(block_conv, 1, "local")
block_local_C = sch.cache_write(block_conv, 0, "local")
write_sch(sch, log_path, "cache_related")

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")
sch.compute_inline(block_im2col)
write_sch(sch, log_path, "Im2ColInline")
sch.compute_inline(block_im2col_pad)
sch.compute_inline(block_flat_pad)
sch.compute_inline(block_flat)
write_sch(sch, log_path, "FlatInline")

(i, j, k) = sch.get_loops(block_conv)
bx, tz, i, ty = sch.split(
    i, factors=[None, num_warps, chunk, num_ty])
k, tx, vk, kernel_k = sch.split(k, factors=[None, num_tx, vec // MMA_K, MMA_K])
sch.bind(bx, "blockIdx.x")
sch.bind(tz, "threadIdx.z")
sch.bind(ty, "threadIdx.y")
sch.bind(tx, "threadIdx.x")
write_sch(sch, log_path, "extract_compute")

# cache read A from global memory to shared_memory
sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
# sch.compute_at(block_shared_A, i, preserve_unit_loops=True)
sch.compute_at(block_shared_local_B, tx, preserve_unit_loops=True)
# sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
sch.reverse_compute_at(block_local_C, ty, preserve_unit_loops=True)
write_sch(sch, log_path, "compute_at_related")


A_local_j = sch.get_loops(block_shared_local_A)[-1]
A_local_j, A_local_vec = sch.split(A_local_j, factors=[None, vec])
sch.vectorize(A_local_vec)
write_sch(sch, log_path, "schedule_local_A")

B_local_fused = sch.fuse(*sch.get_loops(block_shared_local_B)[-2:])
B_local_outer, B_local_vec = sch.split(B_local_fused, factors=[None, vec])
sch.vectorize(B_local_vec)
write_sch(sch, log_path, "schedule_local_B")

# sch.decompose_reduction(block_conv, k)
# sch.tensorize(kernel_k, DP4A_INTRIN)

# sch.unroll(vk)
# sch.unroll(sch.get_loops(block_shared_local_A)[-1])
# sch.unroll(sch.get_loops(block_shared_local_B)[-1])


ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# a_np = np.ones(data_shape).astype("float16")
# b_np = np.ones(kernel_shape).astype("float16")
# random init a and b
a_np = (np.random.uniform(size=data_shape) * 5).astype("float16")
b_np = (np.random.uniform(size=kernel_shape) * 5).astype("float16")
# aragnge init a and b
# a_np = np.mod(np.arange(0, batch_size * in_channels * height * width), 4).reshape(data_shape).astype("float16")
# b_np = np.mod(np.arange(0, kernel_h * kernel_w * in_channels * factor), 4).reshape(kernel_shape).astype("float16")

c_np = np.zeros((M, N)).astype(out_dtype)
cuda_a = tvm.nd.array(a_np, ctx)
cuda_b = tvm.nd.array(b_np, ctx)
cuda_c = tvm.nd.array(c_np, ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)


if VERIFY:
    # do conv with torch
    import torch
    # convert a from nhwc to nchw
    a_np = np.transpose(a_np, (1, 2, 3, 0))
    a_np = np.transpose(a_np, (0, 3, 1, 2))
    a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float32)
    # convert b from ohwi to oihw
    b_np = np.transpose(b_np, (0, 3, 1, 2))
    b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float32)
    m = torch.nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(pad_h, pad_w),
                        groups=in_channels, dtype=torch.float32, bias=False).cuda()
    m.weight.data = b_torch.data
    c_torch = m(a_torch)
    print(c_torch.shape)
    c_torch_np = np.transpose(c_torch.detach().cpu().numpy(), (0, 2, 3, 1))
    # nhwc to cnhw
    c_torch_np = np.transpose(c_torch_np, (3, 0, 1, 2))
    c_torch_np = c_torch_np.reshape((M * N))
    print("torch result: ", c_torch_np[0:-10])
    print("tvm result: ", cuda_c.asnumpy().reshape((M * N))[0:-10])
    # print("verify result: ", np.allclose(c_torch_np, cuda_c.asnumpy()))

num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

print("convert conv2d into M %d N %d K %d 's gemm, average time cost of %d runs = %g ms" %
      (M, N, K, num_runs, t * 1e3))
