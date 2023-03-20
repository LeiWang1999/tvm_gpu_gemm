"""
Problem definition:
    1.We read a float16 matrix A of shape (M, K) from global memory to shared memory.
    2.We need to do permutation on A to make it suitable for dp4a conflict free access.
    3.So we first need to read A from global memory to local memory.
    4.Then we need to do permutation on A in local memory.
    5.Finally we need to read A from local memory to shared memory.
Solution:
    In this python code, we use tensorir transform layout to do permutation.
    Take a Gemm example, and the size of Gemm is a Wrap tile of nvidia cutlass, which is 128x128x16.
Result:
    average time cost of 1 runs = 136.129 ms, 64616.1 GFLOPS. Sota Implementation!
"""

import tvm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tvm import te
import numpy as np
import tvm.testing
from tvm.script import tir as T
from intrin.DP4A import DP4A_INTRIN
import os

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/tensorirscript_simt/gemv_hfma2/" + fname
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

M = 2
N = 1
K = 1

MMA_M = 2
MMA_N = 1
MMA_K = 1


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M // MMA_M, K, MMA_M], dtype="float16")
        B = T.match_buffer(b, [N, K], dtype="float16")
        C = T.match_buffer(c, [M // MMA_M, N, MMA_M], dtype="float16")

        for i, j, k, ii in T.grid(M // MMA_M, N, K, MMA_M):
            with T.block("B"):
                vi, vj, vk, vii = T.axis.remap("SSRS", [i, j, k, ii])
                with T.init():
                    C[vi, vj, vii] = 0.0
                C[vi, vj, vii] = C[vi, vj, vii] + \
                    A[vi, vk, vii].astype("float16") * B[vj, vk].astype("float16")


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(ir_module.script())

block_b = sch.get_block("B")
i, j, k, kernel_i = sch.get_loops(block_b)
# block_shared_A = sch.cache_read(block_b, 0, "shared")
block_shared_local_A = sch.cache_read(block_b, 0, "local")
# block_shared_B = sch.cache_read(block_b, 1, "shared")
block_shared_local_B = sch.cache_read(block_b, 1, "local")
block_local_C = sch.cache_write(block_b, 0, "local")
write_sch(sch, log_path, "cache_related")

# sch.reorder(i, j, k, kernel_k, kernel_i, kernel_j)
sch.bind(i, "blockIdx.x")
sch.bind(j, "threadIdx.x")
write_sch(sch, log_path, "do_split")

# cache read A from global memory to shared_memory
sch.compute_at(block_shared_local_A, k)
# sch.compute_at(block_shared_A, k)
sch.compute_at(block_shared_local_B, k)
# sch.compute_at(block_shared_B, k)
sch.reverse_compute_at(block_local_C, j)

# sch.vectorize(kernel_i)
# write_sch(sch, log_path, "do_tensorize")

sch.decompose_reduction(block_b, k)
write_sch(sch, log_path, "decompose_reduction")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M // MMA_M, K, MMA_M)).astype("float16"), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((N, K)).astype("float16"), ctx)
cuda_c = tvm.nd.array(np.zeros((M // MMA_M, N, MMA_M)).astype("float16"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
