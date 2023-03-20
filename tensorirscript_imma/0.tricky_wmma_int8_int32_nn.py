import tvm
from tvm import te
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from tvm.tir.tensor_intrin.cuda import (
    WMMA_FILL_16x16x16_S32_INTRIN,
    WMMA_LOAD_16x16x16_S8_A_INTRIN,
    WMMA_LOAD_16x16x16_S8_B_INTRIN,
    WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN,
    WMMA_SYNC_16x16x16_s8s8s32_INTRIN,
    WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN,
    WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN,
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


M = 16384
N = 16384
K = 16384
warp_size = 32
block_row_warps = 1
block_col_warps = 4
warp_row_tiles = 16
warp_col_tiles = 1
chunk = 2
vec = 16
wmma_m = 16
wmma_n = 16
wmma_k = 16
split_k = 16

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M // wmma_m, K // wmma_k, wmma_m, wmma_k], dtype="int8")
        B = T.match_buffer(b, [K // wmma_k, N // wmma_n,
                           wmma_k, wmma_n], dtype="int8")
        C = T.match_buffer(c, [M // wmma_m, N // wmma_n, wmma_m, wmma_n], dtype="int32")

        for ii, jj, kk, i, j, k  in T.grid(M // wmma_m, N // wmma_n, K // wmma_k, wmma_m, wmma_n, wmma_k):
            with T.block("B"):
                vii, vjj, vkk, vi, vj, vk = T.axis.remap("SSRSSR", [ii, jj, kk, i, j, k])
                with T.init():
                    C[vii, vjj, vi, vj] = 0
                C[vii, vjj, vi, vj] = C[vii, vjj, vi, vj] + \
                    A[vii, vkk, vi, vk].astype(
                        "int32") * B[vkk, vjj, vk, vj].astype("int32")


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(type(ir_module))
print(ir_module.script())

write_sch(sch, log_path, "original")
block_b = sch.get_block("B")
block_shared_A = sch.cache_read(block_b, 0, "shared")
block_shared_local_A = sch.cache_read(block_b, 0, "wmma.matrix_a")
block_shared_B = sch.cache_read(block_b, 1, "shared")
block_shared_local_B = sch.cache_read(block_b, 1, "wmma.matrix_b")
block_local_C = sch.cache_write(block_b, 0, "wmma.accumulator")

write_sch(sch, log_path, "cache_related")

(i, j, k, kernel_i, kernel_j, kernel_k) = sch.get_loops(block_b)
block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)
if split_k > 0:
    block_k, block_j = sch.split(block_j, factors=[None, split_k])

write_sch(sch, log_path, "block_tile")

if split_k > 0:
    sch.bind(block_k, "blockIdx.z")
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
A_shared_fused = sch.fuse(*sch.get_loops(block_shared_A)[-4:])
A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
    A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
sch.vectorize(A_shared_vi)
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
sch.bind(A_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_A_shared")

B_shared_fused = sch.fuse(*sch.get_loops(block_shared_B)[-4:])
B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
    B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
sch.vectorize(B_shared_vi)
sch.bind(B_shared_tx, "threadIdx.x")
sch.bind(B_shared_ty, "threadIdx.y")
sch.bind(B_shared_tz, "threadIdx.z")

write_sch(sch, log_path, "schedule_B_shared")

# decompose reduction
init_block_b = sch.decompose_reduction(block_b, ko)
write_sch(sch, log_path, "decompose_reduction")

sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_S32_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
sch.tensorize(sch.get_loops(block_shared_local_A)[-2], WMMA_LOAD_16x16x16_S8_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
sch.tensorize(sch.get_loops(block_shared_local_B)
              [-2], WMMA_LOAD_16x16x16_S8_B_INTRIN)
sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_s8s8s32_INTRIN)
sch.tensorize(sch.get_loops(block_local_C)
              [-2], WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN)
write_sch(sch, log_path,
           "tensorize")

# unroll
write_sch(sch, log_path,
           "do_unroll")


ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M // wmma_m, K // wmma_k, wmma_m, wmma_k)).astype("int8"), ctx)
cuda_b = tvm.nd.array(np.arange(
    N * K).reshape((K // wmma_k,  N // wmma_n, wmma_k,  wmma_n)).astype("int8"), ctx)
cuda_c = tvm.nd.array(np.zeros((M // wmma_m, N // wmma_m, wmma_m, wmma_n)).astype("int32"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
