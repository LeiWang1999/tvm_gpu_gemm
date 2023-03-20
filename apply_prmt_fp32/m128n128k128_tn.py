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

# each warp produces 
block_row_warps = 2
block_col_warps = 4
warp_row_tiles = 4
warp_col_tiles = 2
chunk = 1

# single block and single warps.
wmma_m = 16
wmma_n = 16
wmma_k = 8

warp_size = 32
vec = 4

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [K // wmma_k, M // wmma_m, wmma_k, wmma_m])
        B = T.match_buffer(b, [K // wmma_k, N // wmma_n, wmma_k, wmma_n])
        C = T.match_buffer(c, [M // wmma_m, N // wmma_n, wmma_m, wmma_n])

        for ii, jj, kk, i, j, k  in T.grid(M // wmma_m, N // wmma_n, K // wmma_k, wmma_m, wmma_n, wmma_k):
            with T.block("B"):
                vii, vjj, vkk, vi, vj, vk = T.axis.remap("SSRSSR", [ii, jj, kk, i, j, k])
                with T.init():
                    C[vii, vjj, vi, vj] = 0.0
                C[vii, vjj, vi, vj] = C[vii, vjj, vi, vj] + \
                    A[vkk, vii, vk, vi] * B[vkk, vjj, vk, vj]

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


i, j, k, kernel_i, kernel_j, kernel_k= sch.get_loops(block_b)
mma_i_ty, mma_i= sch.split(kernel_i, factors=[8, wmma_m // 8])
mma_j_tx, mma_j = sch.split(kernel_j, factors=[4, wmma_n // 4])
block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, mma_i_ty, mma_j_tx, ko, ki, kernel_k, ii,
            jj, mma_i, mma_j)
mma_warp = sch.fuse(mma_i_ty, mma_j_tx)

sch.bind(block_i, "blockIdx.x")
sch.bind(block_j, "blockIdx.y")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.z")
sch.bind(mma_warp, "threadIdx.x")

write_sch(sch, log_path, "bind_block")


sch.compute_at(block_local_A, kernel_k, preserve_unit_loops = True)
sch.compute_at(block_local_B, kernel_k, preserve_unit_loops = True)
sch.compute_at(block_shared_A, ko, preserve_unit_loops = True)
sch.compute_at(block_shared_B, ko, preserve_unit_loops = True)
sch.reverse_compute_at(block_cl, mma_warp, preserve_unit_loops = True)
write_sch(sch, log_path, "compute_at")

# schedule shared A
block_shared_A_fused_i_j = sch.fuse(*sch.get_loops(block_shared_A)[-3:])
A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
    block_shared_A_fused_i_j, factors=[block_row_warps, block_col_warps, None, warp_size, vec])

sch.vectorize(A_shared_vi)
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
sch.bind(A_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_A_shared")

# schedule shared B
block_shared_B_fused_i_j = sch.fuse(*sch.get_loops(block_shared_B)[-4:])
B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
    block_shared_B_fused_i_j, factors=[block_row_warps, block_col_warps, None, warp_size, vec])

sch.vectorize(B_shared_vi)
sch.bind(B_shared_tx, "threadIdx.x")
sch.bind(B_shared_ty, "threadIdx.y")
sch.bind(B_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_B_shared")

# schedule c local
block_local_c_i, block_local_c_j = sch.get_loops(block_cl)[-2:]
block_c_fused = sch.fuse(block_local_c_i, block_local_c_j)
_, block_c_vi = sch.split(block_c_fused, [None, vec])
sch.vectorize(block_c_vi)
write_sch(sch, log_path, "schedule_C_local")

# schedule A local
# block_local_A_i, block_local_A_j = sch.get_loops(block_local_A)[-2:]
# block_A_fused = sch.fuse(block_local_A_i, block_local_A_j)
# _, block_A_vi = sch.split(block_A_fused, [None, vec])
# sch.vectorize(block_A_vi)
write_sch(sch, log_path, "schedule_A_local")

# schedule B local
# block_local_B_i, block_local_B_j = sch.get_loops(block_local_B)[-2:]
# block_B_fused = sch.fuse(block_local_B_i, block_local_B_j)
# _, block_B_vi = sch.split(block_B_fused, [None, vec])
# sch.vectorize(block_B_vi)
write_sch(sch, log_path, "schedule_B_local")

# sch.vectorize(sch.get_loops(block_local_A)[-1])
# sch.vectorize(sch.get_loops(block_local_B)[-1])

init_block = sch.decompose_reduction(block_b, ko)
write_sch(sch, log_path, "decompose_reduction")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")
print(cuda_mod.imported_modules[0].get_source())


cuda_a = tvm.nd.array(np.arange(M * K).reshape((K // wmma_k, M // wmma_m, wmma_k, wmma_m)).astype("float32"), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((K // wmma_k, N // wmma_n, wmma_k, wmma_n)).astype("float32"), ctx)
cuda_c = tvm.nd.array(np.zeros((M // wmma_m, N // wmma_n, wmma_m, wmma_n)).astype("float32"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
