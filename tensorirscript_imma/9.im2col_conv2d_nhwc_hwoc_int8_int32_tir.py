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
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
    WMMA_FILL_16x16x16_F16_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN
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


# The sizes of inputs and filters
batch_size = 1
height = 224
width = 224
in_channels = 256
out_channels = 512
kernel_h = 7
kernel_w = 7
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
block_row_warps = 1
block_col_warps = 1
warp_row_tiles = 1
warp_col_tiles = 1
warp_size = 32
chunk = 4
vec = 8
split_k = 16

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size,
    height,
    width,
    in_channels
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    kernel_h,
    kernel_w,
    in_channels,
    out_channels
)
# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size,
    output_height,
    output_width,
    out_channels,
)
M = output_height * output_width
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
    def func(A: T.Buffer[(batch_size, height, width, in_channels), "float16"], W: T.Buffer[(kernel_h, kernel_w, in_channels, out_channels), "float16"], Conv: T.Buffer[(batch_size, M, N), "float16"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        Apad = T.alloc_buffer([batch_size, 226, 226, 256], dtype="float16")
        data_im2col = T.alloc_buffer([batch_size, 48400, 12544], dtype="float16")
        weight_flatten = T.alloc_buffer([12544, 512], dtype="float16")
        data_im2colPad = T.alloc_buffer([batch_size, MPAD, KPAD], dtype="float16")
        weight_flattenPad = T.alloc_buffer([KPAD, NPAD], dtype="float16")
        CPad = T.alloc_buffer([batch_size, MPAD, NPAD], dtype="float32")

        for n, h, w, i in T.grid(batch_size, 226, 226, 256):
            with T.block("Apad"):
                v_n, v_h, v_w, v_i = T.axis.remap("SSSS", [n, h, w, i])
                T.reads(A[v_n, v_h - 1, v_w - 1, v_i])
                T.writes(Apad[v_n, v_h, v_w, v_i])
                Apad[v_n, v_h, v_w, v_i] = T.if_then_else(1 <= v_h and v_h < 225 and 1 <= v_w and v_w < 225, A[v_n, v_h - 1, v_w - 1, v_i], T.float16(0), dtype="float16")
        for n, x, y in T.grid(1, 48400, 12544):
            with T.block("data_im2col"):
                v_n, v_x, v_y = T.axis.remap("SSS", [n, x, y])
                T.reads(Apad[v_n, v_y // 1792 + v_x // 220, v_y % 1792 // 256 + v_x % 220, v_y % 256])
                T.writes(data_im2col[v_n, v_x, v_y])
                data_im2col[v_n, v_x, v_y] = Apad[v_n, v_y // 1792 + v_x // 220, v_y % 1792 // 256 + v_x % 220, v_y % 256]
        for x, y in T.grid(12544, 512):
            with T.block("weight_flatten"):
                v_x, v_y = T.axis.remap("SS", [x, y])
                T.reads(W[v_x // 1792, v_x % 1792 // 256, v_x % 256, v_y])
                T.writes(weight_flatten[v_x, v_y])
                weight_flatten[v_x, v_y] = W[v_x // 1792, v_x % 1792 // 256, v_x % 256, v_y]
        
        for n, i, k in T.grid(batch_size, MPAD, KPAD):
            with T.block("data_im2colPad"):
                vn, vi, vk = T.axis.remap("SSS", [n, i, k])
                data_im2colPad[vn, vi, vk] = T.if_then_else(vi < M and vk < K, data_im2col[vn, vi, vk], T.float16(0), dtype="float16")
        
        for k, j in T.grid(KPAD, NPAD):
            with T.block("weight_flattenPad"):
                vk, vj = T.axis.remap("SS", [k, j])
                weight_flattenPad[vk, vj] = T.if_then_else(vk < K and vj < N, weight_flatten[vk, vj], T.float16(0), dtype="float16")
            
        
        for n, x, y, k in T.grid(1, 48400, 512, 12544):
            with T.block("Conv"):
                v_n, v_x, v_y, v_k = T.axis.remap("SSSR", [n, x, y, k])
                T.reads(data_im2col[v_n, v_x, v_k], weight_flatten[v_k, v_y])
                T.writes(Conv[v_n, v_x, v_y])
                with T.init():
                    Conv[v_n, v_x, v_y] = T.float16(0)
                Conv[v_n, v_x, v_y] = Conv[v_n, v_x, v_y] + data_im2col[v_n, v_x, v_k] * weight_flatten[v_k, v_y]

ir_module = MyModule

print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("Apad")
block_im2col = sch.get_block("data_im2col")
block_flat = sch.get_block("weight_flatten")
block_conv = sch.get_block("Conv")
block_conv_input_global = sch.cache_read(block_conv, 0 ,"global")
block_conv_input_shared = sch.cache_read(block_conv, 0 ,"shared")
block_conv_input_frag = sch.cache_read(block_conv, 0, "wmma.matrix_a")
block_conv_weight_global = sch.cache_read(block_conv, 1 ,"global")
block_conv_weight_shared = sch.cache_read(block_conv, 1 ,"shared")
block_conv_weight_frag = sch.cache_read(block_conv, 1, "wmma.matrix_b")
block_conv_output_frag = sch.cache_write(block_conv, 0, "wmma.accumulator")
write_sch(sch, log_path, "cache_related")

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")
sch.compute_inline(block_im2col)
write_sch(sch, log_path, "Im2ColInline")
sch.compute_inline(block_flat)
write_sch(sch, log_path, "FlatInline")

def tricky_transform_A(n, i, j):
    return (n, i // wmma_m, j // wmma_k, i % wmma_m, j % wmma_k)


def tricky_transform_B(i, j):
    return (i // wmma_k, j // wmma_n, i % wmma_k, j % wmma_n)


def tricky_transform_C(n, i, j):
    return (n, i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n)

sch.transform_layout(block_conv_input_global, ("write", 0),tricky_transform_A)
sch.transform_layout(block_conv_weight_global, ("write", 0),tricky_transform_B)
sch.transform_layout(block_conv_input_shared, ("write", 0), tricky_transform_A)
sch.transform_layout(block_conv_weight_shared, ("write", 0), tricky_transform_B)
sch.transform_layout(block_conv_input_frag, ("write", 0), tricky_transform_A)
sch.transform_layout(block_conv_weight_frag, ("write", 0), tricky_transform_B)
sch.transform_layout(block_conv, ("write", 0), tricky_transform_C)

write_sch(sch, log_path, "transform_layout")

(n, i, j, k) = sch.get_loops(block_conv)
i, kernel_i = sch.split(i, factors=[None, wmma_m])
j, kernel_j = sch.split(j, factors=[None, wmma_n])
k, kernel_k = sch.split(k, factors=[None, wmma_k])
sch.reorder(i, j, k, kernel_i, kernel_j, kernel_k)

write_sch(sch, log_path, "tricky_extract_compute")

block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)
block_k, block_j = sch.split(block_j, factors=[None, split_k])

write_sch(sch, log_path, "block_tile")

sch.bind(block_k, "blockIdx.z")
sch.bind(block_i, "blockIdx.y")
sch.bind(block_j, "blockIdx.x")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.z")

write_sch(sch, log_path, "thread_bind")

# cache read A from global memory to shared_memory
sch.compute_at(block_conv_input_frag, ki)
sch.compute_at(block_conv_input_shared, ko)
sch.compute_at(block_conv_weight_frag, ki)
sch.compute_at(block_conv_weight_shared, ko)
sch.reverse_compute_at(block_conv_output_frag, j)
write_sch(sch, log_path, "cache_read_compute_at")

def tricky_extract_cache(block, sub_i, sub_j):
    i, j = sch.get_loops(block)[-2:]
    i, kernel_i = sch.split(i, factors=[None, sub_i])
    j, kernel_j = sch.split(j, factors=[None, sub_j])
    sch.reorder(i, j, kernel_i, kernel_j)
    return (i, j, kernel_i, kernel_j)


block_conv_input_frag_loops = tricky_extract_cache(
    block_conv_input_frag, wmma_m, wmma_k)
block_conv_input_shared_loops = tricky_extract_cache(
    block_conv_input_shared, wmma_m, wmma_k)
block_conv_weight_frag_loops = tricky_extract_cache(
    block_conv_weight_frag, wmma_n, wmma_k)
block_conv_weight_shared_loops = tricky_extract_cache(
    block_conv_weight_shared, wmma_n, wmma_k)
block_conv_output_frag_loops = tricky_extract_cache(
    block_conv_output_frag, wmma_m, wmma_n)

write_sch(sch, log_path, "tricky_extract_cache")

# 128x32
A_shared_fused = sch.fuse(*sch.get_loops(block_conv_input_shared)[-4:])
A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
    A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
sch.vectorize(A_shared_vi)
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

sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_F16_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
sch.tensorize(sch.get_loops(block_conv_input_frag)[-2], WMMA_LOAD_16x16x16_F16_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
sch.tensorize(sch.get_loops(block_conv_weight_frag)[-2], WMMA_LOAD_16x16x16_F16_B_INTRIN)
sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_INTRIN)
sch.tensorize(sch.get_loops(block_conv_output_frag)
              [-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)
write_sch(sch, log_path,
           "tensorize")

# unroll
write_sch(sch, log_path,
           "do_unroll")

def schedule_tricky_transform(block, vec):
    i, j = sch.get_loops(block)[-2:]
    if K <= 16384:
        fused_axis = sch.fuse(i, j)
        # 16384
        by, bx, vx, ty, tx, fused_inner, fused_vi = sch.split(
            fused_axis, factors=[4, 2048, 1, 128, 8, None, vec])
        # 8192
        # by, bx, vx, ty, tx, fused_inner, fused_vi = sch.split(
        #     fused_axis, factors=[256, 256, 4, 2, 8, None, vec])
        
        sch.vectorize(fused_vi)
        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        # sch.unroll(fused_inner)
    else:
        bx, fused_inner, ty, tx, fused_vi = sch.split(
            j, factors=[1024, None, 32, 32, vec])
        sch.vectorize(fused_vi)
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

schedule_tricky_transform(block_conv_input_global, vec=vec)
schedule_tricky_transform(block_conv_weight_global, vec=vec)

write_sch(sch, log_path, "schedule_tricky_transform")



ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

a_np = np.ones(data_shape).astype("float16")
b_np = np.ones(kernel_shape).astype("float16")
c_np = np.zeros((batch_size, M, N)).astype("float16")
cuda_a = tvm.nd.array(a_np, ctx)
cuda_b = tvm.nd.array(b_np, ctx)
cuda_c = tvm.nd.array(c_np, ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

print("convert conv2d into B %d M %d N %d K %d 's gemm, average time cost of %d runs = %g ms" %
      (batch_size, M, N, K, num_runs, t * 1e3))
