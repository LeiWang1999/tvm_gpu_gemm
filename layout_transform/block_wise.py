# NT Layout Transform: Block Wise Pattern

import tvm
from tvm.script import tir as T
import numpy as np


def shared_load_16x16_to_A_global_16x16_layout(i, j):
    # 0, 0-7 -> 0, 0-7
    # 1, 0-7 -> 1, 0-7
    # 2, 0-7 -> 2, 0-7
    # 3, 0-7 -> 3, 0-7

    thread_id = i + (j // 8) * 16
    row = thread_id // 2
    col = (thread_id % 2) * 8 + (j % 8)
    return row, col

# NN Layout Transform: Vector Wise Pattern


def shared_load_16x16_to_B_global_16x16_layout(i, j):
    # 0, 0-7 -> 0, 0-7
    # 1, 0-7 -> 1, 0-7
    # 2, 0-7 -> 2, 0-7

    thread_id = (i % 8) + (j // 8) * 8 + ((i // 8) % 2) * 64
    row = thread_id // 2
    col = (thread_id % 2) * 8 + (j % 8)
    return row, col


M = 16384
N = 16384
wmma_m = 16
wmma_n = 16
dtype = "float16"
vec = 8
warp_size = 32
warp_nums = 4
chunk = 2

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
sch.bind(bx, "blockIdx.x")
sch.bind(inner_j, "blockIdx.y")
sch.bind(ty, "threadIdx.y")
sch.bind(tx, "threadIdx.x")

print(sch.mod)

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
print(cuda_mod.imported_modules[0].get_source())
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
