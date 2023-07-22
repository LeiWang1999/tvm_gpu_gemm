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
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from intrin.tricky_mma_float16_float16 import (
    TRICKY_MMA_fill_16x16_f16_INTRIN,
    TRICKY_LDMATRIX_16x16_A_INTRIN,
    TRICKY_LDMATRIX_16x16_B_INTRIN,
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
    TRICKY_MMA_f16f16f16_INTRIN,
    TRICKY_MMA_f16f16f16_TRANS_INTRIN,
    TRICKY_MMA_store_16x16_f16_global_INTRIN,
    TRICKY_MMA_store_16x16_f16_shared_INTRIN,
    A_global_16x16_to_shared_load_16x16_layout,
    B_global_16x16_to_shared_load_16x16_layout,
    shared_load_16x16_to_A_global_16x16_layout,
    shared_load_16x16_to_B_global_16x16_layout,
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
batch_size = 128
height = 28
width = 28
in_channels = 128
out_channels = 128
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
warp_row_tiles = 8
warp_col_tiles = 2
chunk = 2
raster = 32
stage = 3

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
    out_channels // wmma_n,
    kernel_h,
    kernel_w,
    in_channels // wmma_k,
    wmma_n,
    wmma_k,
)
# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size // wmma_m,
    output_height,
    output_width,
    out_channels // wmma_n,
    wmma_m,
    wmma_n,
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

print("M: ", M, " MPAD: ", MPAD)
print("N: ", N, " NPAD: ", NPAD)
print("K: ", K, " KPAD: ", KPAD)

# Algorithm
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, w: T.handle, conv: T.handle):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        A = T.match_buffer(a, data_shape, dtype="float16")
        W = T.match_buffer(w, (N // wmma_m, K // wmma_k, wmma_m, wmma_k), dtype="float16")
        A_permutate = T.alloc_buffer((batch_size // wmma_m, height, width, in_channels // wmma_k, wmma_m, wmma_k), dtype="float16")
        Apad = T.alloc_buffer([batch_size // wmma_m, height + 2*pad_h, width + 2*pad_w, in_channels // wmma_k, wmma_m, wmma_k], dtype="float16")
        Conv = T.match_buffer(conv, (M // wmma_m, N // wmma_n, wmma_m, wmma_n), dtype="float16")
        data_im2col = T.alloc_buffer([M // wmma_m, K // wmma_k, wmma_m, wmma_k], dtype="float16")

        for n, h, _w, c, nn, cc in T.grid(batch_size // wmma_m, height, width, in_channels // wmma_k, wmma_m, wmma_k):
            with T.block("A_permutate"):
                v_n, v_h, v__w, v_c, v_nn, v_cc = T.axis.remap("SSSSSS", [n, h, _w, c, nn, cc])
                A_permutate[v_n, v_h, v__w, v_c, v_nn, v_cc] = A[v_n * wmma_m + v_nn, v_h, v__w, v_c * wmma_k + v_cc]

        for n, h, _w, i, ii, jj in T.grid(batch_size, height + 2*pad_h, width + 2*pad_w, in_channels // wmma_k, wmma_m, wmma_k):
            with T.block("Apad"):
                v_n, v_h, v_w, v_i, vii, vjj = T.axis.remap("SSSSSS", [n, h, _w, i, ii, jj])
                Apad[v_n, v_h, v_w, v_i, vii, vjj] = T.if_then_else(pad_h <= v_h and v_h < height + pad_h and pad_w <= v_w and v_w < width + pad_w, A_permutate[v_n, v_h - pad_h, v_w - pad_w, v_i, vii, vjj], T.float16(0), dtype="float16")

        for x, y, xx, yy in T.grid(M // wmma_m, K // wmma_k, wmma_m, wmma_k):
            with T.block("data_im2col"):
                v_x, v_y, v_xx, v_yy = T.axis.remap("SSSS", [x, y, xx, yy])
                data_im2col[v_x, v_y, v_xx, v_yy] = Apad[
                v_x // (output_height * output_width),
                stride_h * ((v_x % (output_height * output_width)) // output_width) + dilation_h * ((v_y // in_channels) // kernel_w),
                stride_w * ((v_x % (output_height * output_width)) % output_width) + dilation_w * ((v_y // in_channels) % kernel_w),
                v_y % in_channels,
                v_xx,
                v_yy
            ]

        for xx, yy, kk, x, y, k in T.grid(M // wmma_m, N // wmma_n, K // wmma_k, wmma_m, wmma_n, wmma_k):
            with T.block("Conv"):
                v_xx, v_yy, v_kk, v_x, v_y, v_k = T.axis.remap("SSRSSR", [xx, yy, kk, x, y, k])
                with T.init():
                    Conv[v_xx, v_yy, v_x, v_y] = T.float16(0)
                Conv[v_xx, v_yy, v_x, v_y] = Conv[v_xx, v_yy, v_x, v_y] + data_im2col[v_xx, v_kk, v_x, v_k] * W[v_yy, v_kk, v_y, v_k]


ir_module = MyModule

print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_a_permutate = sch.get_block("A_permutate")
block_pad = sch.get_block("Apad")
block_im2col = sch.get_block("data_im2col")
block_conv = sch.get_block("Conv")
block_conv_input_shared = sch.cache_read(block_conv, 0 ,"shared")
block_conv_input_frag = sch.cache_read(block_conv, 0, "warp")
block_conv_weight_shared = sch.cache_read(block_conv, 1 ,"shared")
block_conv_weight_frag = sch.cache_read(block_conv, 1, "warp")
block_conv_output_frag = sch.cache_write(block_conv, 0, "warp")
write_sch(sch, log_path, "cache_related")

def A_permutation(n, h, w, c):
    kernel_i = n % wmma_m
    kernel_j = c % wmma_k
    new_kernel_i, new_kernel_j = A_global_16x16_to_shared_load_16x16_layout(kernel_i, kernel_j)
    # convert (n // wmma_m, h, w, c // wmma_k, new_kernel_i, new_kernel_j) to (n, h, w, c)
    return ((n // wmma_m)* wmma_m + new_kernel_i, h, w, (c // wmma_k)* wmma_k + new_kernel_j)
    
def B_permutation(n, k, kernel_i, kernel_j):
    return (n, k, *B_global_16x16_to_shared_load_16x16_layout(kernel_i, kernel_j))

# sch.transform_layout(block_a_permutate, ("read", 0),
#                      A_permutation)



sch.compute_inline(block_a_permutate)
write_sch(sch, log_path, "PermutateInline")
sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")
sch.compute_inline(block_im2col)
write_sch(sch, log_path, "Im2ColInline")

# sch.transform_layout(block_conv_weight_shared, ("read", 0),
#                      B_permutation)

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

def A_permutation_write(n, k, kernel_i, kernel_j):
    new_kernel_i, new_kernel_j = shared_load_16x16_to_A_global_16x16_layout(kernel_i, kernel_j)
    return (n, k, new_kernel_i, new_kernel_j)

sch.transform_layout(block_conv_input_shared, ("write", 0),
                     A_permutation_write, rewrite_type=2)

block_conv_input_frag_loops = tricky_extract_cache(
    block_conv_input_frag, wmma_m, wmma_k)

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
init_block_b_loops = sch.get_loops(init_block_b)



def index_map_A(i, k, wmma_m, wmma_k):
    return (i, k, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_k))


def index_map_B(h, w, n, c, wmma_n, wmma_k):
    return (h, w, n, c, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_n, wmma_k),)


def index_map_C(m, n, wmma_m, wmma_n):
    return (m, n, *C_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_n),)


sch.transform_layout(block_conv_input_frag, ("write", 0), index_map_A)
sch.transform_layout(block_conv_weight_frag, ("write", 0), index_map_A)
sch.transform_layout(block_conv_output_frag, ("read", 0), index_map_C)

write_sch(sch, log_path, "transform_layout")

sch.tensorize(sch.get_loops(init_block_b)
            [-2], TRICKY_MMA_fill_16x16_f16_INTRIN)
write_sch(sch, log_path, "tensorize_wmma_fill")


sch.tensorize(sch.get_loops(block_conv_input_frag)
            [-2], TRICKY_LDMATRIX_16x16_A_INTRIN)
sch.tensorize(sch.get_loops(block_conv_weight_frag)
            [-2], TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN)
write_sch(sch, log_path, "tensorize_ldmatrix")

sch.tensorize(kernel_i, TRICKY_MMA_f16f16f16_TRANS_INTRIN)
write_sch(sch, log_path, "tensorize_wmma_sync")

sch.tensorize(sch.get_loops(block_conv_output_frag)
            [-2], TRICKY_MMA_store_16x16_f16_global_INTRIN)
write_sch(sch, log_path, "tensorize_store")

# unroll
write_sch(sch, log_path,
        "do_unroll")

if stage > 0:
    sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
    sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
    sch.annotate(ko, "software_pipeline_async_stages", [0])
if raster > 0:
    sch.annotate(init_block_b_loops[-4], ann_key="thread_rasterization", ann_val=raster)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    # code = code.replace("#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1", "#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0")
    if stage == 1:
        code = code.replace(
            '''__asm__ __volatile__("cp.async.commit_group;");''', ' ')
        code = code.replace(
            '''__asm__ __volatile__("cp.async.wait_group 0;");''', '''__asm__ __volatile__("cp.async.commit_group;");
            __asm__ __volatile__("cp.async.wait_group 0;");''')
    # if the next line is a __syncthreads(), replace it with number
    return code
ctx = tvm.cuda(0)
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    cuda_mod = tvm.build(sch.mod, target="cuda")
write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# random init a and b
a_np = np.random.uniform(size=data_shape).astype("float16")
b_np = np.random.uniform(size=(N // wmma_m, K // wmma_k, wmma_n, wmma_k)).astype("float16")
# aragnge init a and b
# a_np = np.mod(np.arange(0, batch_size * in_channels * height * width), 10).reshape(data_shape).astype("float16")
# b_np = np.mod(np.arange(0, kernel_h * kernel_w * in_channels * out_channels), 10).reshape(kernel_shape).astype("float16")

c_np = np.zeros((M // wmma_m, N // wmma_n, wmma_m, wmma_n)).astype("float16")
cuda_a = tvm.nd.array(a_np, ctx)
cuda_b = tvm.nd.array(b_np, ctx)
cuda_c = tvm.nd.array(c_np, ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)


if VERIFY:
    # do conv with torch
    import torch
    # convert a from nhwc1616 into nhwc
    a_np = np.transpose(a_np, (0, 4, 1, 2, 3, 5)).reshape(
        (batch_size, height, width, in_channels))
    # convert a from nhwc to nchw
    a_np = np.transpose(a_np, (0, 3, 1, 2))
    a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float32)
    a_torch = torch.nn.functional.pad(a_torch, (pad_h, pad_h, pad_w, pad_w))

    # convert b from ohwi1616 into ohwi
    b_np = b_np.reshape(kernel_shape)
    b_np = np.transpose(b_np, (0, 4, 1, 2, 3, 5)).reshape(
        (out_channels, kernel_h, kernel_w, in_channels))
    # convert b from ohwi to oihw
    b_np = np.transpose(b_np, (0, 3, 1, 2))
    b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float32)
    c_torch = torch.nn.functional.conv2d(
        a_torch, b_torch, stride=(stride_h, stride_w), groups=1)
    c_torch_np = np.transpose(c_torch.cpu().numpy(), (0, 2, 3, 1))
    print(c_torch_np.shape)

    # convert c from nhwc1616 into nhwc
    c_np = cuda_c.numpy().reshape(output_shape).transpose((0, 4, 1, 2, 3, 5)).reshape(
        (batch_size, output_height, output_width, out_channels)
    )
    print("torch result: ", c_torch_np[0][0][0][0:10])
    print("tvm result: ", c_np[0][0][0][0:10])
    np.testing.assert_allclose(c_torch_np, c_np, atol=1e-1, rtol=1e-1, verbose=True)



num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

print("convert conv2d into B %d M %d N %d K %d 's gemm, average time cost of %d runs = %g ms" %
    (batch_size, M, N, K, num_runs, t * 1e3))
