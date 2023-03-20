from asyncore import write
import tvm
from tvm import te
import os
from tvm.contrib import nvcc
from tvm.contrib import spirv
import numpy as np
import tvm.testing
from tvm.script import tir as T

_dtype = "float32"


log_path = "progress/tensorir_script/4.vectorize"
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


M = N = K = 16384

Grid_Size_X = 128
Grid_Size_Y = 128
Block_Size_X = 16
Block_Size_Y = 16
V_Thread_X = 2
V_Thread_Y = 2

BK = 16


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        AT = T.match_buffer(a, [K, M])
        # A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [K, N])
        C = T.match_buffer(c, [M, N])

        for i, j, k in T.grid(M, K, N):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                # C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
                C[vi, vj] = C[vi, vj] + AT[vk, vi] * B[vk, vj]


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module)

print(type(ir_module))
print(ir_module.script())


'''
read_buffer_index : 0->A 1->B
'''
block_b = sch.get_block("B")
block_shared_A = sch.cache_read(block_b, 0, "shared")
block_local_A = sch.cache_read(block_b, 0, "local")
block_shared_B = sch.cache_read(block_b, 1, "shared")
block_local_B = sch.cache_read(block_b, 1, "local")
block_cl = sch.cache_write(block_b, 0, "local")

write_code(sch.mod["main"].script(), log_path, "0.origin.py")

(i, j, k) = sch.get_loops(block_b)
bx, xi = sch.split(i, factors=[Grid_Size_X, None])
by, yi = sch.split(j, factors=[Grid_Size_Y, None])
sch.reorder(bx, by, xi, yi)
write_code(sch.mod["main"].script(), log_path, "1.reorder.py")
sch.bind(by, "blockIdx.y")
sch.bind(bx, "blockIdx.x")

tyz, yi = sch.split(yi, factors=[V_Thread_Y, None])
ty, yi = sch.split(yi, [Block_Size_Y, None])
txz, xi = sch.split(xi, factors=[V_Thread_X, None])
tx, xi = sch.split(xi, [Block_Size_X, None])
sch.reorder(tyz, txz, ty, tx, xi, yi)
sch.bind(ty, "threadIdx.y")
sch.bind(tx, "threadIdx.x")
sch.bind(tyz, "vthread.y")
sch.bind(txz, "vthread.x")
write_code(sch.mod["main"].script(), log_path, "2.thread_bind.py")


sch.reverse_compute_at(block_cl, tx, preserve_unit_loops=True)
write_code(sch.mod["main"].script(), log_path, "3.cache_write_compute_at.py")

ko, ki = sch.split(k, [None, BK])
write_code(sch.mod["main"].script(), log_path, "4.split.py")

sch.reorder(ko, ki, yi, xi)

sch.compute_at(block_local_A, ki)
sch.compute_at(block_local_B, ki)
sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)

'''
sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
sch.compute_at(block_local_A, ki)
sch.compute_at(block_local_B, ki)

can not run because:
 The primitive requires all the consumer(s) of the given block to be present under the target loop. However, there are 1 consumer(s) not satisfying the constraint. List of the consumer(s):tir.Block#0
'''


write_code(sch.mod["main"].script(), log_path, "4.cache_read_compute_at.py")

aa_yi, aa_xi = sch.get_loops(block_shared_A)[-2:]  # loops size is 7
aa_yi, aa_ty = sch.split(aa_yi, factors=[None, Block_Size_Y])
aa_xi, aa_tx = sch.split(aa_xi, factors=[None, Block_Size_X * 4])
aa_tx, aa_vi = sch.split(aa_tx, factors=[Block_Size_X, None])
sch.reorder(aa_ty, aa_tx, aa_yi, aa_xi, aa_vi)
sch.bind(aa_ty, "threadIdx.y")
sch.bind(aa_tx, "threadIdx.x")

loops = sch.get_loops(block_shared_B)
bb_yi, bb_xi = sch.get_loops(block_shared_B)[-2:]
bb_yi, bb_ty = sch.split(bb_yi, factors=[None, Block_Size_Y])
bb_xi, bb_tx = sch.split(bb_xi, factors=[None, Block_Size_X * 4])
bb_tx, bb_vi = sch.split(bb_tx, factors=[Block_Size_X, None])
sch.reorder(bb_ty, bb_tx, bb_yi, bb_xi, bb_vi)
sch.bind(bb_ty, "threadIdx.y")
sch.bind(bb_tx, "threadIdx.x")

sch.decompose_reduction(block_b, ko)

sch.vectorize(aa_vi)
sch.vectorize(bb_vi)

sch.vectorize(sch.get_loops(block_local_A)[-1])
sch.vectorize(sch.get_loops(block_local_B)[-1])

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.py")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype(_dtype), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype(_dtype), ctx)
cuda_c = tvm.nd.array(np.zeros((M, N)).astype(_dtype), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 30
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
