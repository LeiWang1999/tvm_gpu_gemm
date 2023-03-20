from asyncore import write
import tvm
from tvm import te
import os
from tvm.contrib import nvcc
from tvm.contrib import spirv
import numpy as np
import tvm.testing
from tvm.script import tir as T

log_path = "progress/tensorir_script/cutlass_tile_nn"
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


M = N = K = 16384

# tuning params
block_row_warps = 16
block_col_warps = 16
vthread_row_tiles = 2
vthread_col_tiles = 2
thread_row_tiles = 4
thread_col_tiles = 4
raster = 1
chunk = 16
vec = 4

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [K, N])
        C = T.match_buffer(c, [M, N])

        for i, j, k in T.grid(M, K, N):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

ir_module = MyModule
sch = tvm.tir.Schedule(ir_module)

print(type(ir_module))
print(ir_module.script())


'''
read_buffer_index : 0->A 1->B
'''
block_b = sch.get_block("B")
block_permute_A = sch.cache_read(block_b, 0, "local")
block_shared_A = sch.cache_read(block_b, 0, "shared")
block_local_A = sch.cache_read(block_b, 0, "local")
block_shared_B = sch.cache_read(block_b, 1, "shared")
block_local_B = sch.cache_read(block_b, 1, "local")
block_cl = sch.cache_write(block_b, 0, "local")

write_sch(sch, log_path, "origin")


(i, j, k) = sch.get_loops(block_b)
block_i, vi, i, ii = sch.split(i, factors=[None, vthread_row_tiles, block_row_warps,  thread_row_tiles])
block_j, vj, j, jj = sch.split(j, factors=[None, vthread_col_tiles, block_col_warps, thread_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, vi, vj, i, j, ko, ki, ii, jj)

write_sch(sch, log_path, "block_tile")

sch.bind(block_i, "blockIdx.y")
sch.bind(block_j, "blockIdx.x")
sch.bind(vi, "vthread.y")
sch.bind(vj, "vthread.x")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.x")

write_sch(sch, log_path, "thread_bind")

sch.reverse_compute_at(block_cl, j)
write_sch(sch, log_path, "cache_write_compute_at")


# sch.transform_layout(block_local_A, ("write", 0), lambda i,k: (k,i) )

write_sch(sch, log_path, "transform_layout")

sch.compute_at(block_local_A, ki)
sch.compute_at(block_local_B, ki)
sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
sch.compute_at(block_permute_A, ko)

write_sch(sch, log_path, "cache_read_compute_at")

block_permute_A_i, block_permute_A_j = sch.get_loops(block_permute_A)[-2:]
block_permute_A_fuse = sch.fuse(block_permute_A_i, block_permute_A_j)
block_permute_A_outer, block_permute_A_tx, block_permute_A_ty = sch.split(block_permute_A_fuse, factors=[None, block_col_warps, block_row_warps])

sch.bind(block_permute_A_tx, "threadIdx.x")
sch.bind(block_permute_A_ty, "threadIdx.y")
write_sch(sch, log_path, "schedule_A_permute")

sch.reverse_compute_at(block_shared_A, block_permute_A_ty)
sch.transform_layout(block_shared_A, ("write", 0), lambda i,k: (k,i))
write_sch(sch, log_path, "schedule_A_shared")

# sch.vectorize(sch.get_loops(block_permute_A)[-1])

B_shared_fused = sch.fuse(*sch.get_loops(block_shared_B)[-2:])
B_shared_outer, B_shared_ty, B_shared_tx, B_shared_vi = sch.split(
    B_shared_fused, factors=[None, block_row_warps, block_col_warps, vec])
sch.vectorize(B_shared_vi)
sch.bind(B_shared_tx, "threadIdx.x")
sch.bind(B_shared_ty, "threadIdx.y")
write_sch(sch, log_path, "schedule_B_shared")

block_b_init = sch.decompose_reduction(block_b, ko)
write_sch(sch, log_path, "decompose_reduction")

block_local_A_outer, block_local_A_vi = sch.split(sch.get_loops(block_local_A)[-1], factors=[None, vec])
sch.vectorize(block_local_A_vi)
block_local_B_outer, block_local_B_vi = sch.split(sch.get_loops(block_local_B)[-1], factors=[None, vec])
sch.vectorize(block_local_B_vi)

sch.annotate(ko, ann_key="thread_rasterization", ann_val=raster)

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype("float32"), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype("float32"), ctx)
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
