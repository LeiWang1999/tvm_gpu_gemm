import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from intrin.tricky_mma_float16_float16 import (
    TRICKY_MMA_fill_16x16_f16_INTRIN,
    TRICKY_LDMATRIX_16x16_A_INTRIN,
    TRICKY_LDMATRIX_16x16_A_INTRIN_DYN,
    TRICKY_LDMATRIX_16x16_B_INTRIN,
    TRICKY_LDMATRIX_16x16_B_INTRIN_DYN,
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN_DYN,
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
    TRICKY_MMA_f16f16f16_INTRIN,
    TRICKY_MMA_f16f16f16_TRANS_INTRIN,
    TRICKY_MMA_store_16x16_f16_global_INTRIN,
    A_global_16x16_to_shared_load_16x16_layout,
    C_shared_16x16_to_ldmatrix_32x8_layout,
    A_B_shared_16x16_to_ldmatrix_32x8_layout
)

from intrin.async_copy import ASYNC_COPY_F16_X8_INTRIN_DYN

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

M = 16384
N = 16384
K = 16384
if VERIFY:
    M = 256
    N = 2048
    K = 1024

warp_size = 32
# nni search results:
block_row_warps = 1
block_col_warps = 4
warp_row_tiles = 8
warp_col_tiles = 4
chunk = 2
raster = 8
stage = 1  # 1 is no double buffer 2 is double buffer enabled

vec = 8
wmma_m = 16
wmma_n = 16
wmma_k = 16


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M // wmma_m, K // wmma_k,
                           wmma_m, wmma_k], dtype="float16")
        B = T.match_buffer(b, [K // wmma_k, N // wmma_n,
                           wmma_k, wmma_n], dtype="float16")
        C = T.match_buffer(c, [M // wmma_m, N // wmma_n,
                           wmma_m, wmma_n], dtype="float16")

        for ii, jj, kk, i, j, k in T.grid(M // wmma_m, N // wmma_n, K // wmma_k, wmma_m, wmma_n, wmma_k):
            with T.block("B"):
                vii, vjj, vkk, vi, vj, vk = T.axis.remap(
                    "SSRSSR", [ii, jj, kk, i, j, k])
                with T.init():
                    C[vii, vjj, vi, vj] = 0.0
                C[vii, vjj, vi, vj] = C[vii, vjj, vi, vj] + \
                    A[vii, vkk, vi, vk] * B[vkk, vjj, vk, vj]


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(type(ir_module))
print(ir_module.script())

write_sch(sch, log_path, "original")
block_b = sch.get_block("B")
block_shared_A = sch.cache_read(block_b, 0, "shared.dyn")
block_shared_local_A = sch.cache_read(block_b, 0, "warp")
block_shared_B = sch.cache_read(block_b, 1, "shared.dyn")
block_shared_local_B = sch.cache_read(block_b, 1, "warp")
block_local_C = sch.cache_write(block_b, 0, "warp")

write_sch(sch, log_path, "cache_related")

(i, j, k, kernel_i, kernel_j, kernel_k) = sch.get_loops(block_b)
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
sch.compute_at(block_shared_local_A, ki)
sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
sch.compute_at(block_shared_local_B, ki)
sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
sch.reverse_compute_at(block_local_C, j)
write_sch(sch, log_path, "cache_read_compute_at")


# 128x32
def permutation(i, j, kernel_i, kernel_j):
    return (i, j, *A_global_16x16_to_shared_load_16x16_layout(kernel_i, kernel_j))


sch.transform_layout(block_shared_A, ("read", 0),
                     permutation)
sch.transform_layout(block_shared_B, ("read", 0),
                     permutation)

write_sch(sch, log_path, "transform_shared_read_layout")

A_shared_ii, A_shared_jj, A_shared_i, A_shared_j = sch.get_loops(
    block_shared_A)[-4:]
A_shared_j, A_shared_vi = sch.split(A_shared_j, factors=[None, vec])
sch.vectorize(A_shared_vi)
A_shared_fused = sch.fuse(A_shared_ii, A_shared_jj, A_shared_i, A_shared_j)
A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx = sch.split(
    A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size])
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
sch.bind(A_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_A_shared")

B_shared_ii, B_shared_jj, B_shared_i, B_shared_j = sch.get_loops(
    block_shared_B)[-4:]
B_shared_j, B_shared_vi = sch.split(B_shared_j, factors=[None, vec])
sch.vectorize(B_shared_vi)
B_shared_fused = sch.fuse(B_shared_ii, B_shared_jj, B_shared_i, B_shared_j)
B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx = sch.split(
    B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size])
sch.bind(B_shared_tx, "threadIdx.x")
sch.bind(B_shared_ty, "threadIdx.y")
sch.bind(B_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_B_shared")

# decompose reduction
init_block_b = sch.decompose_reduction(block_b, ko)
init_block_b_loops = sch.get_loops(init_block_b)
write_sch(sch, log_path, "decompose_reduction")

# transpose layout


def index_map_A(i, k, wmma_m, wmma_k):
    return (i, k, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_k))


def index_map_B(j, k, wmma_n, wmma_k):
    return (j, k, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_n, wmma_k),)


def index_map_C(i, j, wmma_m, wmma_n):
    return (i, j, *C_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_n),)


sch.transform_layout(block_shared_local_A, ("write", 0), index_map_A)
sch.transform_layout(block_shared_local_B, ("write", 0), index_map_A)
sch.transform_layout(block_local_C, ("read", 0), index_map_C)
write_sch(sch, log_path, "transform_layout")

init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
sch.tensorize(sch.get_loops(init_block_b)
              [-2], TRICKY_MMA_fill_16x16_f16_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(
    block_shared_local_A)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_A)
              [-2], TRICKY_LDMATRIX_16x16_A_INTRIN_DYN)
write_sch(sch, log_path,
          "tensorize_load")
block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(
    block_shared_local_B)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_B)
              [-2], TRICKY_LDMATRIX_16x16_B_INTRIN_DYN)
sch.tensorize(kernel_i, TRICKY_MMA_f16f16f16_INTRIN)

sch.tensorize(sch.get_loops(block_local_C)
              [-2], TRICKY_MMA_store_16x16_f16_global_INTRIN)
write_sch(sch, log_path,
          "tensorize")

# unroll
# sch.unroll(init_block_b_i)
# sch.unroll(init_block_b_j)
# sch.unroll(block_shared_local_A_i)
# sch.unroll(block_shared_local_A_j)
# sch.unroll(block_shared_local_B_i)
# sch.unroll(block_shared_local_B_j)
# sch.unroll(ki)
# sch.unroll(ko)
# sch.unroll(ii)
# sch.unroll(jj)
# sch.unroll(A_shared_inner)
# sch.unroll(B_shared_inner)

if stage > 1:

    sch.annotate(ki, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
    sch.annotate(ki, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
    
    sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, stage - 1, stage - 1])
    sch.annotate(ko, ann_key="software_pipeline_order",
                 ann_val=[0, 1, 3, 2, 4])
    # sch.annotate(ko, ann_key="software_pipeline_async_stages", ann_val=[0])


if raster > 0:
    sch.annotate(init_block_b_loops[-4],
                 ann_key="thread_rasterization", ann_val=raster)

write_sch(sch, log_path,
          "do_unroll")


# @tvm.register_func
# def tvm_callback_cuda_postproc(code):
#     code_lines = code.split("\n")
#     # Keep track of the current wait group number
#     commit_count = 0
#     for i, line in enumerate(code_lines):
#         if '__asm__ __volatile__("cp.async.wait_group' in line: 
#             del code_lines[i]

#     for i, line in enumerate(code_lines):
#         # line = code_lines[i]
#         if '__asm__ __volatile__("cp.async.commit_group;");' in line:
#             # If a line contains the __asm__ __volatile__("cp.async.commit_group;") statement, increment the counter
#             commit_count += 1
#             if commit_count < stage:
#                 del code_lines[i]      
#             else:
#                 # insert a wait group statement
#                 code_lines.insert(i+1, '__asm__ __volatile__("cp.async.wait_group 0;");')
#                 break
#     # Re-assemble the code string
#     new_code = "\n".join(code_lines)
#     return new_code

ctx = tvm.cuda(0)
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    cuda_mod = tvm.build(sch.mod, target="cuda")

    
write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

a_np = (np.random.rand
        (M // wmma_m, K // wmma_k, wmma_m, wmma_k)).astype("float16")

b_np = (np.random.rand
        (K // wmma_k, N // wmma_n, wmma_k, wmma_n)).astype("float16")
# a_np = np.mod(np.arange(M * K).reshape(M // wmma_m, K // wmma_k, wmma_m, wmma_k), 4).astype("float16")
# b_np = np.mod(np.arange(N * K).reshape(K // wmma_k, N // wmma_n, wmma_k, wmma_n), 5).astype("float16")
cuda_a = tvm.nd.array((a_np).astype("float16"), ctx)
cuda_b = tvm.nd.array((b_np).astype("float16"), ctx)
cuda_c = tvm.nd.array(
    np.zeros((M // wmma_m, N // wmma_m, wmma_m, wmma_n)).astype("float16"), ctx)

if VERIFY:
    cuda_mod(cuda_a, cuda_b, cuda_c)
    a_np = a_np.transpose((0, 2, 1, 3)).reshape(M, K)
    b_np = b_np.transpose((0, 2, 1, 3)).reshape(K, N)
    c_np = cuda_c.numpy().transpose((0, 2, 1, 3)).reshape(M, N)
    np_c = np.matmul(a_np.astype("float16"), b_np.astype("float16"))
    print("np result: ", np_c[0][0:10])
    print("tvm result: ", c_np[0][0:10])
    np.testing.assert_allclose(
        c_np, np_c, rtol=1e-2, atol=1e-2
    )
# cuda_mod(cuda_a, cuda_b, cuda_c)
# print(cuda_c.numpy())
for i in range(0, 5):
    cuda_mod(cuda_a, cuda_b, cuda_c)
num_flops = 2 * M * K * N
num_runs = 5
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
