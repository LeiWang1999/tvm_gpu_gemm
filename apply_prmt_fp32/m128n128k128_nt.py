from asyncore import write
import tvm
from tvm import te
import os
from tvm.contrib import nvcc
from tvm.contrib import spirv
import numpy as np
import tvm.testing
from tvm.script import tir as T

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/apply_prmt_fp32/" + fname
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


M = 128
N = 128
K = 128

# single block and single warps.
MMA_M = 16
MMA_N = 16
MMA_K = 8

warp_size = 32
vec = 4


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [N, K])
        C = T.match_buffer(c, [M, N])

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

ir_module = MyModule
sch = tvm.tir.Schedule(ir_module)
                                                                                                                                                                
print(type(ir_module))
print(ir_module.script())
write_sch(sch, log_path, "origin")

block_b = sch.get_block("B")
block_shared_A = sch.cache_read(block_b, 0, "shared")
block_local_A = sch.cache_read(block_b, 0, "local")
block_shared_B = sch.cache_read(block_b, 1, "shared")
block_local_B = sch.cache_read(block_b, 1, "local")
block_cl = sch.cache_write(block_b, 0, "local")
write_sch(sch, log_path, "cache_related")


i, j, k = sch.get_loops(block_b)
i, mma_i = sch.split(i, factors=[None, MMA_M])
j, mma_j = sch.split(j, factors=[None, MMA_N])
k, mma_k = sch.split(k, factors=[None, MMA_K])
sch.reorder(i, j, k, mma_k, mma_i, mma_j)
mma_fused_i_j = sch.fuse(mma_i, mma_j)
mma_fused_i_j, mma_tx = sch.split(mma_fused_i_j, factors=[None, warp_size])
sch.bind(i, "blockIdx.y")
sch.bind(j, "blockIdx.x")
sch.bind(mma_tx, "threadIdx.x")

write_sch(sch, log_path, "bind_block")


sch.compute_at(block_local_A, mma_k)
sch.compute_at(block_local_B, mma_k)
sch.compute_at(block_shared_A, k)
sch.compute_at(block_shared_B, k)
sch.reverse_compute_at(block_cl, j)
write_sch(sch, log_path, "compute_at")

# schedule shared A
block_shared_A_i, block_shared_A_j = sch.get_loops(block_shared_A)[-2:]
block_shared_A_fused_i_j = sch.fuse(block_shared_A_i, block_shared_A_j)
block_shared_A_outer, block_shared_A_tx, block_shared_A_vi = sch.split(block_shared_A_fused_i_j, factors=[None, warp_size, vec])
sch.vectorize(block_shared_A_vi)
sch.bind(block_shared_A_tx, "threadIdx.x")

# schedule shared B
block_shared_B_i, block_shared_B_j = sch.get_loops(block_shared_B)[-2:]
block_shared_B_fused_i_j = sch.fuse(block_shared_B_i, block_shared_B_j)
block_shared_B_outer, block_shared_B_tx, block_shared_B_vj = sch.split(block_shared_B_fused_i_j, factors=[None, warp_size, vec])
sch.vectorize(block_shared_B_vj)
sch.bind(block_shared_B_tx, "threadIdx.x")

# schedule local C
block_cl_i, block_cl_j = sch.get_loops(block_cl)[-2:]
block_cl_fused_i_j = sch.fuse(block_cl_i, block_cl_j)
block_cl_outer, block_cl_tx, block_cl_vj = sch.split(block_cl_fused_i_j, factors=[None, warp_size, vec])
sch.vectorize(block_cl_vj)
sch.bind(block_cl_tx, "threadIdx.x")

init_block = sch.decompose_reduction(block_b, k)
init_block_i, init_block_j = sch.get_loops(init_block)[-2:]
sch.bind(init_block_j, "threadIdx.x")
write_sch(sch, log_path, "decompose_reduction")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")
print(cuda_mod.imported_modules[0].get_source())


cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype("float32"), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((N, K)).astype("float32"), ctx)
cuda_c = tvm.nd.array(np.zeros((M, N)).astype("float32"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
