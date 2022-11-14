"""
Problem definition:
    1.We read a int8 matrix A of shape (M, K) from global memory to shared memory.
    2.We need to do permutation on A to make it suitable for dp4a conflict free access.
    3.So we first need to read A from global memory to local memory.
    4.Then we need to do permutation on A in local memory.
    5.Finally we need to read A from local memory to shared memory.
Solution:
    In this python code, we use tensorir transform layout to do permutation.
    Take a Gemm example, and the size of Gemm is a Wrap tile of nvidia cutlass, which is 128x128x16.
"""

import tvm
from tvm import te
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from tvm.tir import TensorIntrin

log_path = "progress/dp4a_permutation/transform_layout"
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


M = 256
N = 256
K = 256
BM = 128
BN = 128
BK = 32
TX = 8
TY = 8


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="int8")
        B = T.match_buffer(b, [K, N], dtype="int8")
        C = T.match_buffer(c, [M, N], dtype="int32")

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("int32") * B[vk, vj].astype("int32")


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(type(ir_module))
print(ir_module.script())

write_code(sch.mod.astext(), log_path, "original.cu")
block_b = sch.get_block("B")
block_shared_A = sch.cache_read(block_b, 0, "shared")
block_local_B = sch.cache_read(block_b, 1, "local")
block_local_permutation_B = sch.cache_read(block_b, 1, "local")
block_local_permutation_shared_B = sch.cache_read(block_b, 1, "shared")
block_local_C = sch.cache_write(block_b, 0, "local")

write_code(sch.mod.astext(), log_path, "cache_related.cu")

(i, j, k) = sch.get_loops(block_b)
bx, i = sch.split(i, factors=[None, BM])
by, j = sch.split(j, factors=[None, BN])
bk, k = sch.split(k, factors=[None, BK])

i, tm = sch.split(i, factors=[None, 8])
j, tn = sch.split(j, factors=[None, 8])
k, vk = sch.split(k, factors=[None, 4])
write_code(sch.mod.astext(), log_path, "split_inner_loops.cu")

sch.reorder(bx, by, bk, i, j, tm, tn, k, vk)
write_code(sch.mod.astext(), log_path, "reorder_inner_loops.cu")

sch.bind(bx, "blockIdx.x")
sch.bind(by, "blockIdx.y")
sch.bind(i, "threadIdx.x")
sch.bind(j, "threadIdx.y")

write_code(sch.mod.astext(), log_path, "thread_bind.cu")

# cache read A from global memory to shared_memory
sch.compute_at(block_shared_A, bk)
sch.compute_at(block_local_permutation_shared_B, bk)
sch.compute_at(block_local_permutation_B, bk)
sch.compute_at(block_local_B, bk)
sch.reverse_compute_at(block_local_C, j)

write_code(sch.mod.astext(), log_path, "cache_read_compute_at.cu")


A_shared_fused = sch.fuse(*sch.get_loops(block_shared_A)[-2:])
A_shared_ty, A_shared_tx, A_shared_inner = sch.split(
    A_shared_fused, factors=[16, 16, None])
sch.vectorize(A_shared_inner)
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
write_code(sch.mod.astext(), log_path, "schedule_A_shared.cu")

# do transformation
# def B_local_transformation(m, n):
#     # 32 * 128
#     return (m // 4, n // 4, m % 4, n % 4)

# sch.transform_layout(block_local_B, ("write", 0), B_local_transformation)
# write_code(sch.mod.astext(), log_path, "transform_B_local.cu")
B_local_i, B_local_j = sch.get_loops(block_local_B)[-2:]
# 32x128
B_local_oi, B_local_i = sch.split(B_local_i, factors=[8, 4])
B_local_oj, B_local_j = sch.split(B_local_j, factors=[32, 4])
sch.reorder(B_local_oi, B_local_oj, B_local_i, B_local_j)
B_local_fused = sch.fuse(B_local_oi, B_local_oj)
B_local_tx, B_local_ty = sch.split(B_local_fused, factors=[16, 16])
# sch.vectorize(B_local_j)
sch.bind(B_local_tx, "threadIdx.x")
sch.bind(B_local_ty, "threadIdx.y")
write_code(sch.mod.astext(), log_path, "schedule_B_local.cu")

# do permutation


def B_local_permutation(m, n):
    # 32 * 128
    start_pos_m = (m // 4) * 4
    start_pos_n = (n // 4) * 4
    temp_m = n % 4
    temp_n = m % 4
    return (start_pos_m + temp_m, start_pos_n + temp_n)


sch.transform_layout(block_local_permutation_B,
                     ("write", 0), B_local_permutation)

write_code(sch.mod.astext(), log_path, "transform_B_local_permutation.cu")
B_local_permutation_i, B_local_permutation_j = sch.get_loops(
    block_local_permutation_B)[-2:]
# 32x128
B_local_permutation_oi, B_local_permutation_i = sch.split(
    B_local_permutation_i, factors=[8, 4])
B_local_permutation_oj, B_local_permutation_j = sch.split(
    B_local_permutation_j, factors=[32, 4])
sch.reorder(B_local_permutation_oi, B_local_permutation_oj,
            B_local_permutation_i, B_local_permutation_j)
B_local_permutation_fused = sch.fuse(
    B_local_permutation_oi, B_local_permutation_oj)
B_local_permutation_tx, B_local_permutation_ty = sch.split(
    B_local_permutation_fused, factors=[16, 16])
sch.bind(B_local_permutation_tx, "threadIdx.x")
sch.bind(B_local_permutation_ty, "threadIdx.y")
write_code(sch.mod.astext(), log_path, "schedule_B_local_permutation.cu")

# schedule shared
sch.transform_layout(block_local_permutation_shared_B,
                     ("write", 0), B_local_permutation)

B_local_permutation_shared_i, B_local_permutation_shared_j = sch.get_loops(
    block_local_permutation_shared_B)[-2:]
write_code(sch.mod.astext(), log_path,
           "transform_B_local_permutation_shared.cu")
# 32x128
B_local_permutation_shared_oi, B_local_permutation_shared_i = sch.split(
    B_local_permutation_shared_i, factors=[8, 4])
B_local_permutation_shared_oj, B_local_permutation_shared_j = sch.split(
    B_local_permutation_shared_j, factors=[32, 4])
sch.reorder(B_local_permutation_shared_oi, B_local_permutation_shared_oj,
            B_local_permutation_shared_i, B_local_permutation_shared_j)
B_local_permutation_shared_fused = sch.fuse(
    B_local_permutation_shared_oi, B_local_permutation_shared_oj)
B_local_permutation_shared_tx, B_local_permutation_shared_ty = sch.split(
    B_local_permutation_shared_fused, factors=[16, 16])
sch.bind(B_local_permutation_shared_tx, "threadIdx.x")
sch.bind(B_local_permutation_shared_ty, "threadIdx.y")
B_local_permutation_shared_inner_fused = sch.fuse(
    B_local_permutation_shared_i, B_local_permutation_shared_j)
write_code(sch.mod.astext(), log_path,
           "schedule_B_local_permutation_shared.cu")

# decompose reduction
init_block_b = sch.decompose_reduction(block_b, bk)
write_code(sch.mod.astext(), log_path, "decompose_reduction.cu")
sch.bind(sch.get_loops(init_block_b)[2], "threadIdx.x")
sch.bind(sch.get_loops(init_block_b)[3], "threadIdx.y")

# dp4a tensorize


@T.prim_func
def dp4a_desc(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    B: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4], B[0:4])
        T.writes(C[0])
        for i in range(0, 4):
            with T.block("update"):
                vi = T.axis.remap("R", [i])
                C[0] = C[0] + T.cast(A[vi], "int32") * T.cast(B[vi], "int32")


@T.prim_func
def dp4a_impl(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    B: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4], B[0:4])
        T.writes(C[0])

        C[0] += T.call_pure_extern(
            "__dp4a", A.vload([0], "int8x4"), B.vload([0], "int8x4"), T.int32(0), dtype="int32"
        )


DP4A_INTRIN = "my_dp4a"

TensorIntrin.register(DP4A_INTRIN, dp4a_desc, dp4a_impl)
sch.mod.show()
sch.tensorize(vk, DP4A_INTRIN)

write_code(sch.mod.astext(), log_path,
           "do_dp4a_tensorize.cu")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype("int8"), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype("int8"), ctx)
cuda_c = tvm.nd.array(np.zeros((M, N)).astype("int32"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 10
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
