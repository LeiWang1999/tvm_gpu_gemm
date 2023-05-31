# NT Layout Transform: Block Wise Pattern

import tvm
from tvm.script import tir as T
import numpy as np


def A_global_16x16_to_shared_load_16x16_layout(i, j):
    thread_id = i * 2 + j // 8
    row = thread_id % 16
    col = (j % 8) + (thread_id // 16) * 8
    return row, col


def B_global_16x16_to_shared_load_16x16_layout(i, j):
    thread_id = i * 2 + j // 8
    row = (i // 8) * 8 + (thread_id % 8)
    col = (j % 8) + 8 * ((thread_id // 8) % 2)
    return row, col


M = 32
N = 1024
wmma_m = 16
wmma_n = 16
dtype = "float16"
vec = 2
warp_size = 32
warp_nums = 4
chunk = 4

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(
            a, (M, N), dtype)
        B = T.match_buffer(
            b, (M // wmma_m, N // wmma_n, wmma_m, wmma_n), dtype)

        for i, j in T.grid(M, N):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi // wmma_m, vj // wmma_m, vi % wmma_m, vj % wmma_n] = A[vi, vj]


ir_module = MyModule

print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

block_b = sch.get_block("B")
i, j = sch.get_loops(block_b)
bx, inner_i = sch.split(i, factors=[None, chunk])
inner_j, ty, tx, vj = sch.split(j, factors=[None, warp_nums, warp_size, vec])

sch.vectorize(vj)
sch.bind(bx, "blockIdx.y")
sch.bind(inner_j, "blockIdx.x")
sch.bind(ty, "threadIdx.y")
sch.bind(tx, "threadIdx.x")

def permutation_A(i, j):
    ii = i // 16
    jj = j // 16
    kernel_i = i % 16
    kernel_j = j % 16
    ti, tj, tki, tkj = (ii, jj, *B_global_16x16_to_shared_load_16x16_layout(kernel_i, kernel_j))
    return (ti * 16 + tki, tj * 16 + tkj)

sch.transform_layout(block_b, ("read", 0),
                     permutation_A, rewrite_type=1)

sch.unroll(inner_i)
print(sch.mod)

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
# print(cuda_mod.imported_modules[0].get_source())
cuda_a=tvm.nd.array(np.ones(
    M * N).reshape((M, N)).astype(dtype), ctx)
cuda_b = tvm.nd.array(np.ones(
    M * N).reshape((M // wmma_m, N // wmma_n, wmma_m, wmma_n)).astype(dtype), ctx)
cuda_mod(cuda_a, cuda_b)

num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b).mean

print("average time cost of %d runs = %g ms." %
      (num_runs, t * 1e3))
