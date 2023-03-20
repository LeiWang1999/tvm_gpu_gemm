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
from intrin.tricky_mma_float16_float16 import (
    TRICKY_MMA_fill_16x16_f16_INTRIN,
    TRICKY_LDMATRIX_16x16_A_INTRIN,
    TRICKY_LDMATRIX_16x16_B_INTRIN,
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
    TRICKY_MMA_f16f16f16_INTRIN,
    TRICKY_MMA_f16f16f16_TRANS_INTRIN,
    TRICKY_MMA_store_16x16_f16_global_INTRIN,
    A_global_16x16_to_shared_load_16x16_layout,
    shared_load_16x16_to_A_global_16x16_layout,
    B_global_16x16_to_shared_load_16x16_layout,
    C_shared_16x16_to_ldmatrix_32x8_layout,
    A_B_shared_16x16_to_ldmatrix_32x8_layout
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
batch_size = 2
height = 16
width = 16
in_channels = 1280
out_channels = 1280
kernel_h = 3
kernel_w = 3
pad_h = 1
pad_w = 1
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
block_row_warps = 1
block_col_warps = 4
warp_row_tiles = 4
warp_col_tiles = 2
chunk = 2
raster = 1
stage = 4

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
# padding MPAD as the multiple of block_row_warps * warp_row_tiles * wmma_m
MPAD = (M + block_row_warps * warp_row_tiles * wmma_m - 1) // (
    block_row_warps * warp_row_tiles * wmma_m
) * block_row_warps * warp_row_tiles * wmma_m
# padding NPAD as the multiple of block_col_warps * warp_col_tiles * wmma_n
NPAD = (N + block_col_warps * warp_col_tiles * wmma_n - 1) // (
    block_col_warps * warp_col_tiles * wmma_n
) * block_col_warps * warp_col_tiles * wmma_n
# padding KPAD as the multiple of block_col_warps * warp_col_tiles * wmma_k
KPAD = (K + block_col_warps * warp_col_tiles * wmma_k - 1) // (
    block_col_warps * warp_col_tiles * wmma_k
) * block_col_warps * warp_col_tiles * wmma_k

print("MPAD: ", MPAD)
print("NPAD: ", NPAD)
print("KPAD: ", KPAD)

# Algorithm
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, w: T.handle, conv: T.handle):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        A = T.match_buffer(a, (batch_size, height, width, in_channels), dtype="float16")
        W = T.match_buffer(w, (N // wmma_m, K // wmma_k, wmma_m, wmma_k), dtype="float16")
        Conv = T.match_buffer(conv, (M // wmma_m, N // wmma_n, wmma_m, wmma_n), dtype="float16")
        Apad = T.alloc_buffer([batch_size, height + 2*pad_h, width + 2*pad_w, in_channels], dtype="float16")
        data_im2col = T.alloc_buffer([M, K], dtype="float16")

        for n, h, w, i in T.grid(batch_size, height + 2*pad_h, width + 2*pad_w, in_channels):
            with T.block("Apad"):
                v_n, v_h, v_w, v_i = T.axis.remap("SSSS", [n, h, w, i])
                Apad[v_n, v_h, v_w, v_i] = T.if_then_else(pad_h <= v_h and v_h < height + pad_h and pad_w <= v_w and v_w < width + pad_w, A[v_n, v_h - pad_h, v_w - pad_w, v_i], T.float16(0), dtype="float16")

        for x, y in T.grid(M, K):
            with T.block("data_im2col"):
                v_x, v_y = T.axis.remap("SS", [x, y])
                data_im2col[v_x, v_y] = Apad[
                v_x // (output_height * output_width),
                stride_h * ((v_x % (output_height * output_width)) // output_width) + dilation_h * ((v_y // in_channels) // kernel_w),
                stride_w * ((v_x % (output_height * output_width)) % output_width) + dilation_w * ((v_y // in_channels) % kernel_w),
                v_y % in_channels,
            ]

        for xx, yy, kk, x, y, k in T.grid(M // wmma_m, N // wmma_n, K // wmma_k, wmma_m, wmma_n, wmma_k):
            with T.block("Conv"):
                v_xx, v_yy, v_kk, v_x, v_y, v_k = T.axis.remap("SSRSSR", [xx, yy, kk, x, y, k])
                with T.init():
                    Conv[v_xx, v_yy, v_x, v_y] = T.float16(0)
                Conv[v_xx, v_yy, v_x, v_y] = Conv[v_xx, v_yy, v_x, v_y] + data_im2col[v_xx * wmma_m + v_x, v_kk * wmma_k + v_k] * W[v_yy, v_kk, v_y, v_k]


ir_module = MyModule

print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("Apad")
block_im2col = sch.get_block("data_im2col")
block_conv = sch.get_block("Conv")
block_conv_input_shared = sch.cache_read(block_conv, 0 ,"shared")
block_conv_input_frag = sch.cache_read(block_conv, 0, "warp")
block_conv_weight_shared = sch.cache_read(block_conv, 1 ,"shared")
block_conv_weight_frag = sch.cache_read(block_conv, 1, "warp")
block_conv_output_frag = sch.cache_write(block_conv, 0, "warp")
write_sch(sch, log_path, "cache_related")

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")
sch.compute_inline(block_im2col)
write_sch(sch, log_path, "Im2ColInline")

def tricky_transform_A(i, j):
    return (i // wmma_m, j // wmma_k, i % wmma_m, j % wmma_k)



sch.transform_layout(block_conv_input_shared, ("write", 0), tricky_transform_A)
sch.transform_layout(block_conv_input_frag, ("write", 0), tricky_transform_A)

write_sch(sch, log_path, "transform_layout")

(i, j, k, kernel_i, kernel_j, kernel_k) = sch.get_loops(block_conv)

block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)

write_sch(sch, log_path, "block_tile")

sch.bind(block_i, "blockIdx.y")
sch.bind(block_j, "blockIdx.x")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.z")
write_sch(sch, log_path, "thread_bind")

# cache read A from global memory to shared_memory
sch.compute_at(block_conv_input_frag, ki, preserve_unit_loops=True)
sch.compute_at(block_conv_input_shared, ko, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_frag, ki, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_shared, ko, preserve_unit_loops=True)
sch.reverse_compute_at(block_conv_output_frag, j, preserve_unit_loops=True)
write_sch(sch, log_path, "cache_read_compute_at")

def tricky_extract_cache(block, sub_i, sub_j):
    i, j = sch.get_loops(block)[-2:]
    i, kernel_i = sch.split(i, factors=[None, sub_i])
    j, kernel_j = sch.split(j, factors=[None, sub_j])
    sch.reorder(i, j, kernel_i, kernel_j)
    return (i, j, kernel_i, kernel_j)


block_conv_input_frag_loops = tricky_extract_cache(
    block_conv_input_frag, wmma_m, wmma_k)

write_sch(sch, log_path, "tricky_extract_cache")

# def permutation(n, h, w, c):
#     # i = n * output_height * output_width + h * output_width + w
#     # j = c + (h // output_height) * in_channels * kernel_w + (w // output_width) * in_channels
#     i = n * output_height * output_width + (h // stride_h) * output_width + (w // stride_w)
#     j = c + ((h // stride_h) // output_height) * in_channels * kernel_w + ((w // stride_w) // output_width) * in_channels
#     kernel_i, kernel_j = A_global_16x16_to_shared_load_16x16_layout(i % wmma_m, j % wmma_k)
#     row = (i // wmma_m) * wmma_m + kernel_i
#     col = (j // wmma_k) * wmma_k + kernel_j
#     return row, col

# def A_permutation(n, h, w, c):
#     i, j = permutation(n, h, w, c)
#     fused_i_j = i * K + j
#     return ((fused_i_j // in_channels) // output_width) // output_height, ((fused_i_j // in_channels) // output_width) % output_height, (fused_i_j // in_channels) % output_width, fused_i_j % in_channels,

def A_permutation(i, j, kernel_i, kernel_j):
    return (i, j, *shared_load_16x16_to_A_global_16x16_layout(kernel_i, kernel_j))

def B_permutation(i, j, kernel_i, kernel_j):
    return (i, j, *B_global_16x16_to_shared_load_16x16_layout(kernel_i, kernel_j))

sch.transform_layout(block_conv_input_shared, ("write", 0), A_permutation, rewrite_type=2)

sch.transform_layout(block_conv_weight_shared, ("read", 0),
                     B_permutation)

# 128x32
A_shared_i, A_shared_j = sch.get_loops(block_conv_input_shared)[-2:]
A_shared_j, A_shared_vi = sch.split(A_shared_j, factors=[None, vec])
# sch.storage_align(block_conv_input_shared, buffer_index=0,
#                   axis=-2, factor=wmma_k, offset=vec)
sch.vectorize(A_shared_vi)
# sch.tensorize(A_shared_vi, ASYNC_COPY_F16_X8_INTRIN)
# sch.annotate(ki, "pragma_commit_wait", "")
A_shared_fused = sch.fuse(A_shared_i, A_shared_j)
A_shared_inner, A_shared_tz, A_shared_ty, A_shared_tx = sch.split(
    A_shared_fused, factors=[None, block_col_warps, block_row_warps,  warp_size])
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
sch.bind(A_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_A_shared")

B_shared_fused = sch.fuse(*sch.get_loops(block_conv_weight_shared)[-4:])
B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
    B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
sch.vectorize(B_shared_vi)
sch.bind(B_shared_tx, "threadIdx.x")
sch.bind(B_shared_ty, "threadIdx.y")
sch.bind(B_shared_tz, "threadIdx.z")

write_sch(sch, log_path, "schedule_B_shared")

# decompose reduction
init_block_b = sch.decompose_reduction(block_conv, ko)
write_sch(sch, log_path, "decompose_reduction")
init_block_b_loops = sch.get_loops(init_block_b)

# transpose layout

def index_map_A(i, k, wmma_m, wmma_k):
    return (i, k, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_k))


def index_map_B(j, k, wmma_n, wmma_k):
    return (j, k, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_n, wmma_k),)


def index_map_C(i, j, wmma_m, wmma_n):
    return (i, j, *C_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_n),)


sch.transform_layout(block_conv_input_frag, ("write", 0), index_map_A)
sch.transform_layout(block_conv_weight_frag, ("write", 0), index_map_A)
sch.transform_layout(block_conv_output_frag, ("read", 0), index_map_C)

sch.tensorize(init_block_b_loops[-2], TRICKY_MMA_fill_16x16_f16_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
sch.tensorize(sch.get_loops(block_conv_input_frag)[-2], TRICKY_LDMATRIX_16x16_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
sch.tensorize(sch.get_loops(block_conv_weight_frag)[-2], TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN)
sch.tensorize(kernel_i, TRICKY_MMA_f16f16f16_TRANS_INTRIN)
sch.tensorize(sch.get_loops(block_conv_output_frag)
              [-2], TRICKY_MMA_store_16x16_f16_global_INTRIN)
write_sch(sch, log_path, 
           "tensorize")

# unroll
# sch.unroll(ko)
write_sch(sch, log_path,
           "do_unroll")
# sch.annotate(ko, "software_pipeline_async_stages", [1])
if stage > 0:
    sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
    sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
    # sch.annotate(ko, "software_pipeline_async_stages", [1, 2])
if raster > 0:
    sch.annotate(init_block_b_loops[-4], ann_key="thread_rasterization", ann_val=raster)

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# init a and b to ones
# a_np = np.ones(data_shape).astype("float16")
# b_np = np.ones((N // wmma_m, K // wmma_k, wmma_n, wmma_k)).astype("float16")# random init a and b
# a_np = np.random.uniform(size=data_shape).astype("float16")
# b_np = np.random.uniform(size=(N // wmma_m, K // wmma_k, wmma_n, wmma_k)).astype("float16")
# aragnge init a and b
a_np = np.mod(np.arange(0, batch_size * in_channels * height * width), 10).reshape(data_shape).astype("float16")
b_np = np.mod(np.arange(0, kernel_h * kernel_w * in_channels * out_channels), 10).reshape((N // wmma_m, K // wmma_k, wmma_n, wmma_k)).astype("float16")

c_np = np.zeros((M // wmma_m, N // wmma_n, wmma_m, wmma_n)).astype("float16")
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
    b_np = b_np.transpose((0, 2, 1, 3)).reshape(kernel_shape)
    b_np = np.transpose(b_np, (0, 3, 1, 2))
    b_torch = torch.tensor(b_np, device="cuda")
    c_torch = torch.nn.functional.conv2d(a_torch, b_torch, groups=1)
    c_torch_np = np.transpose(c_torch.cpu().numpy(), (0, 2, 3, 1))
    c_torch_np = c_torch_np.reshape((M, N))
    c_np = cuda_c.numpy().transpose((0, 2, 1, 3)).reshape(M, N)
    print("torch result: ", c_torch_np[0][0:10])
    print("tvm result: ", c_np[0][0:10])
    print("verify result: ", np.allclose(c_torch_np, c_np))

num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

print("convert conv2d into B %d M %d N %d K %d 's gemm, average time cost of %d runs = %g ms" %
      (batch_size, M, N, K, num_runs, t * 1e3))
