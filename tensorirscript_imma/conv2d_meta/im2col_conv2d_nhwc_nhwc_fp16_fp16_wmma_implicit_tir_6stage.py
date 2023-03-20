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
    WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN
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

# tuning params
warp_size = 32
block_row_warps = 4
block_col_warps = 1
warp_row_tiles = 1
warp_col_tiles = 1
chunk = 4
raster = 8
stage = 2
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
        A = T.match_buffer(a, (batch_size, height, width,
                           in_channels), dtype="float16")
        W = T.match_buffer(
            w, (out_channels, kernel_h, kernel_w, in_channels), dtype="float16")
        Conv = T.match_buffer(conv, (M, N), dtype="float16")
        Apad = T.alloc_buffer(
            [batch_size, height + 2*pad_h, width + 2*pad_w, in_channels], dtype="float16")
        data_im2col = T.alloc_buffer([M, K], dtype="float16")
        weight_flatten = T.alloc_buffer([N, K], dtype="float16")

        for n, h, w, i in T.grid(batch_size, height + 2*pad_h, width + 2*pad_w, in_channels):
            with T.block("Apad"):
                v_n, v_h, v_w, v_i = T.axis.remap("SSSS", [n, h, w, i])
                Apad[v_n, v_h, v_w, v_i] = T.if_then_else(pad_h <= v_h and v_h < height + pad_h and pad_w <=
                                                          v_w and v_w < width + pad_w, A[v_n, v_h - pad_h, v_w - pad_w, v_i], T.float16(0), dtype="float16")

        for x, y in T.grid(M, K):
            with T.block("data_im2col"):
                v_x, v_y = T.axis.remap("SS", [x, y])
                data_im2col[v_x, v_y] = Apad[
                    v_x // (output_height * output_width),
                    stride_h * ((v_x % (output_height * output_width)) // output_width) +
                    dilation_h * ((v_y // in_channels) // kernel_w),
                    stride_w * ((v_x % (output_height * output_width)) %
                                output_width) + dilation_w * ((v_y // in_channels) % kernel_w),
                    v_y % in_channels,
                ]
        for x, y in T.grid(N, K):
            with T.block("weight_flatten"):
                v_n, v_k = T.axis.remap("SS", [x, y])
                weight_flatten[v_n, v_k] = W[v_n, (v_k // in_channels) //
                                             kernel_w, (v_k // in_channels) % kernel_w, v_k % in_channels]

        for x, y, k in T.grid(M, N, K):
            with T.block("Conv"):
                v_x, v_y, v_k = T.axis.remap("SSR", [x, y, k])
                with T.init():
                    Conv[v_x, v_y] = T.float16(0)
                Conv[v_x, v_y] = Conv[v_x, v_y] + \
                    data_im2col[v_x, v_k] * weight_flatten[v_y, v_k]


ir_module = MyModule

print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("Apad")
block_im2col = sch.get_block("data_im2col")
block_flat = sch.get_block("weight_flatten")
block_conv = sch.get_block("Conv")
block_conv_input_local = sch.cache_read(block_conv, 0, "local")
block_conv_input_shared = sch.cache_read(block_conv, 0, "shared.dyn")
block_conv_input_frag = sch.cache_read(block_conv, 0, "wmma.matrix_a")
block_conv_weight_local = sch.cache_read(block_conv, 1, "local")
block_conv_weight_shared = sch.cache_read(block_conv, 1, "shared.dyn")
block_conv_weight_frag = sch.cache_read(block_conv, 1, "wmma.matrix_b")
block_conv_output_frag = sch.cache_write(block_conv, 0, "wmma.accumulator")
write_sch(sch, log_path, "cache_related")

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")
sch.compute_inline(block_im2col)
write_sch(sch, log_path, "Im2ColInline")
sch.compute_inline(block_flat)
write_sch(sch, log_path, "FlatInline")


write_sch(sch, log_path, "transform_layout")

(i, j, k) = sch.get_loops(block_conv)
i, kernel_i = sch.split(i, factors=[None, wmma_m])
j, kernel_j = sch.split(j, factors=[None, wmma_n])
k, kernel_k = sch.split(k, factors=[None, wmma_k])
sch.reorder(i, j, k, kernel_i, kernel_j, kernel_k)

write_sch(sch, log_path, "tricky_extract_compute")

block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, ko, ki, ii,
            jj, kernel_i, kernel_j, kernel_k)

write_sch(sch, log_path, "block_tile")

sch.bind(block_i, "blockIdx.y")
sch.bind(block_j, "blockIdx.x")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.z")
write_sch(sch, log_path, "thread_bind")

# cache read A from global memory to shared_memory
sch.compute_at(block_conv_input_frag, ki, preserve_unit_loops=True)
sch.compute_at(block_conv_input_shared, ko, preserve_unit_loops=True)
sch.compute_at(block_conv_input_local, ko, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_frag, ki, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_shared, ko, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_local, ko, preserve_unit_loops=True)
sch.reverse_compute_at(block_conv_output_frag, j, preserve_unit_loops=True)
write_sch(sch, log_path, "cache_read_compute_at")

A_local_i, A_local_j = sch.get_loops(block_conv_input_local)[-2:]
A_local_j, A_local_vi = sch.split(A_local_j, factors=[None, vec])
sch.vectorize(A_local_vi)
A_local_fused = sch.fuse(A_local_i, A_local_j)
A_local_inner, A_local_tz, A_local_ty, A_local_tx = sch.split(
    A_local_fused, factors=[None, block_col_warps, block_row_warps,  warp_size])
sch.bind(A_local_tx, "threadIdx.x")
sch.bind(A_local_ty, "threadIdx.y")
sch.bind(A_local_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_A_local")

B_local_i, B_local_j = sch.get_loops(block_conv_weight_local)[-2:]
B_local_j, B_local_vi = sch.split(B_local_j, factors=[None, vec])
sch.vectorize(B_local_vi)
B_local_fused = sch.fuse(B_local_i, B_local_j)
B_local_inner, B_local_tz, B_local_ty, B_local_tx = sch.split(
    B_local_fused, factors=[None, block_col_warps, block_row_warps,  warp_size])
sch.bind(B_local_tx, "threadIdx.x")
sch.bind(B_local_ty, "threadIdx.y")
sch.bind(B_local_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_B_local")

A_shared_i, A_shared_j = sch.get_loops(block_conv_input_shared)[-2:]
A_shared_j, A_shared_vi = sch.split(A_shared_j, factors=[None, vec])
sch.vectorize(A_shared_vi)
A_shared_fused = sch.fuse(A_shared_i, A_shared_j)
A_shared_inner, A_shared_tz, A_shared_ty, A_shared_tx = sch.split(
    A_shared_fused, factors=[None, block_col_warps, block_row_warps,  warp_size])
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
sch.bind(A_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_A_shared")

B_shared_i, B_shared_j = sch.get_loops(block_conv_weight_shared)[-2:]
B_shared_j, B_shared_vi = sch.split(B_shared_j, factors=[None, vec])
sch.vectorize(B_shared_vi)
B_shared_fused = sch.fuse(B_shared_i, B_shared_j)
B_shared_inner, B_shared_tz, B_shared_ty, B_shared_tx = sch.split(
    B_shared_fused, factors=[None, block_col_warps, block_row_warps,  warp_size])
sch.bind(B_shared_tx, "threadIdx.x")
sch.bind(B_shared_ty, "threadIdx.y")
sch.bind(B_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_B_shared")

# split wmma loop from block_conv_input_frag and block_conv_weight_frag
input_frag_loop_i, input_frag_loop_j = sch.get_loops(block_conv_input_frag)[-2:]
input_frag_loop_i, input_frag_kernel_i = sch.split(input_frag_loop_i, factors=[None, wmma_m])
input_frag_loop_j, input_frag_kernel_j = sch.split(input_frag_loop_j, factors=[None, wmma_k])
# reorder
sch.reorder(input_frag_loop_i, input_frag_loop_j, input_frag_kernel_i, input_frag_kernel_j)

weight_frag_loop_i, weight_frag_loop_j = sch.get_loops(block_conv_weight_frag)[-2:]
weight_frag_loop_i, weight_frag_kernel_i = sch.split(weight_frag_loop_i, factors=[None, wmma_n])
weight_frag_loop_j, weight_frag_kernel_j = sch.split(weight_frag_loop_j, factors=[None, wmma_k])
# reorder
sch.reorder(weight_frag_loop_i, weight_frag_loop_j, weight_frag_kernel_i, weight_frag_kernel_j)

# split wmma loop from block_conv_output_frag
output_frag_loop_i, output_frag_loop_j = sch.get_loops(block_conv_output_frag)[-2:]
output_frag_loop_i, output_frag_kernel_i = sch.split(output_frag_loop_i, factors=[None, wmma_m])
output_frag_loop_j, output_frag_kernel_j = sch.split(output_frag_loop_j, factors=[None, wmma_n])
# reorder
sch.reorder(output_frag_loop_i, output_frag_loop_j, output_frag_kernel_i, output_frag_kernel_j)


# decompose reduction
init_block_b = sch.decompose_reduction(block_conv, ko)
write_sch(sch, log_path, "decompose_reduction")
init_block_b_loops = sch.get_loops(init_block_b)
sch.tensorize(init_block_b_loops[-2], WMMA_FILL_16x16x16_F16_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
sch.tensorize(input_frag_kernel_i, WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
sch.tensorize(weight_frag_kernel_i, WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN)
sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)
sch.tensorize(sch.get_loops(block_conv_output_frag)
              [-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)
write_sch(sch, log_path,
          "tensorize")

# unroll
# sch.unroll(ko)
write_sch(sch, log_path,
          "do_unroll")

if stage > 0:
    sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
    sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
    sch.annotate(ko, ann_key="software_pipeline_async_stages", ann_val=[0])
    
if raster > 0:
    sch.annotate(init_block_b_loops[-4],
                 ann_key="thread_rasterization", ann_val=raster)

# @tvm.register_func
# def tvm_callback_cuda_postproc(code):
#     code_lines = code.split("\n")
#     # Keep track of the current wait group number
#     '''
#         replace:
# __asm__ __volatile__("cp.async.wait_group x;");

#     __syncthreads();
#         into
# __asm__ __volatile__("cp.async.wait_group x;");
#     '''
    
#     for i in range(len(code_lines)):
#         if "cp.async.wait_group" in code_lines[i]:
#             sync_str = code_lines[i+2]
#             if "__syncthreads();" in sync_str:
#                 code_lines[i+2] = sync_str.replace("__syncthreads();", "")

#     # if the next line is a __syncthreads(), replace it with number
            
            
#     # Re-assemble the code string
#     new_code = "\n".join(code_lines)
#     return new_code

ctx = tvm.cuda(0)
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    cuda_mod = tvm.build(sch.mod, target="cuda")
    
write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# a_np = np.ones(data_shape).astype("float16")
# b_np = np.ones(kernel_shape).astype("float16")
# random init a and b
a_np = np.random.uniform(size=data_shape).astype("float16")
b_np = np.random.uniform(size=kernel_shape).astype("float16")
# aragnge init a and b
# a_np = np.mod(np.arange(0, batch_size * in_channels * height * width), 10).reshape(data_shape).astype("float16")
# b_np = np.mod(np.arange(0, kernel_h * kernel_w * in_channels * out_channels), 10).reshape(kernel_shape).astype("float16")

c_np = np.zeros((M, N)).astype("float16")
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
