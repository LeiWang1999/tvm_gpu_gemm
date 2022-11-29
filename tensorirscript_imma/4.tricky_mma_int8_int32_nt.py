'''
Problem definition:
    the mma version of tricky tensorize
    Consider the following matrix multiplication:
        C = A * B
    where A, B, C are all 2D tensors.
    A is of shape [M, K, 16, 32]
    B is of shape [N, K, 16, 32]
    C is of shape [M, N, 16, 16]
    The tricky part is that the innermost dimension of A and B are contiguous.
    We consider a single kernel of  BM=128, BN=128, BK=64
'''
import tvm
from tvm import te
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from intrin.tricky_mma_int8_int32 import (
    TRICKY_MMA_fill_16x16_i32_INTRIN,
    TRICKY_LDMATRIX_16x32_A_INTRIN,
    TRICKY_LDMATRIX_32x16_B_INTRIN,
    TRICKY_LDMATRIX_16x32_B_TRANS_INTRIN,
    TRICKY_MMA_i8i8i32_INTRIN,
    TRICKY_MMA_i8i8i32_TRANS_INTRIN,
    TRICKY_MMA_store_16x16_i32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
    shared_32x16_to_ldmatrix_32x16_layout,
    shared_16x32_to_ldmatrix_32x16_layout,
    shared_16x32_to_ldmatrix_32x16_permutation,
    global_16x32_to_shared_load_16x32_layout,
)

log_path = "progress/tensorirscript_imma/4.tricky_mma_int8_int32_nt"
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
    N = 256
    K = 256

warp_size = 32
block_row_warps = 2
block_col_warps = 2
warp_row_tiles = 2
warp_col_tiles = 8
chunk = 2
vec = 16
wmma_m = 16
wmma_n = 16
wmma_k = 32

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M // wmma_m, K // wmma_k, wmma_m, wmma_k], dtype="int8")
        B = T.match_buffer(b, [N // wmma_n, K // wmma_k,
                           wmma_n, wmma_k], dtype="int8")
        C = T.match_buffer(c, [M // wmma_m, N // wmma_n,
                           wmma_m, wmma_n], dtype="int32")

        for ii, jj, kk, i, j, k  in T.grid(M // wmma_m, N // wmma_n, K // wmma_k, wmma_m, wmma_n, wmma_k):
            with T.block("B"):
                vii, vjj, vkk, vi, vj, vk = T.axis.remap("SSRSSR", [ii, jj, kk, i, j, k])
                with T.init():
                    C[vii, vjj, vi, vj] = T.int32(0)
                C[vii, vjj, vi, vj] = C[vii, vjj, vi, vj] + \
                    A[vii, vkk, vi, vk].astype("int32") * B[vjj, vkk, vj, vk].astype("int32")


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(type(ir_module))
print(ir_module.script())

write_sch(sch, log_path, "original")
block_b = sch.get_block("B")
block_shared_A = sch.cache_read(block_b, 0, "shared")
block_shared_local_A = sch.cache_read(block_b, 0, "warp")
block_shared_B = sch.cache_read(block_b, 1, "shared")
block_shared_local_B = sch.cache_read(block_b, 1, "warp")
block_local_C = sch.cache_write(block_b, 0, "warp")

write_sch(sch, log_path, "cache_related")

(i, j, k, kernel_i, kernel_j, kernel_k) = sch.get_loops(block_b)
block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)

write_sch(sch, log_path, "block_tile")

sch.bind(block_i, "blockIdx.x")
sch.bind(block_j, "blockIdx.y")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.z")

write_sch(sch, log_path, "thread_bind")


# cache read A from global memory to shared_memory
sch.compute_at(block_shared_local_A, ki)
sch.compute_at(block_shared_A, ko)
sch.compute_at(block_shared_local_B, ki)
sch.compute_at(block_shared_B, ko)
sch.reverse_compute_at(block_local_C, j)
write_sch(sch, log_path, "cache_read_compute_at")


# 128x32
# sch.transform_layout(block_b, ("write", 0), permutation)
def permutation(i, j, kernel_i, kernel_j):
    return (i, j, *global_16x32_to_shared_load_16x32_layout(kernel_i, kernel_j))


sch.transform_layout(block_shared_A, ("read", 0),
                     permutation)
sch.transform_layout(block_shared_B, ("read", 0),
                     permutation)

write_sch(sch, log_path, "transform_layout")

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

# transform layout

def index_map_A(i, k, wmma_m, wmma_k):
    return (i, k, *shared_16x32_to_ldmatrix_32x16_layout(wmma_m, wmma_k), )

def index_map_B(j, k, wmma_n, wmma_k):
    return (j, k, *shared_16x32_to_ldmatrix_32x16_layout(wmma_n, wmma_k), )

def index_map_C(i, j, wmma_m, wmma_n):
    return (i, j, *shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_n), )




sch.transform_layout(block_shared_local_A, ("write", 0), index_map_A)
sch.transform_layout(block_shared_local_B, ("write", 0), index_map_B)
sch.transform_layout(block_local_C, ("read", 0), index_map_C)
write_sch(sch, log_path, "transform_layout")

init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
sch.tensorize(sch.get_loops(init_block_b)[-2], TRICKY_MMA_fill_16x16_i32_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(block_shared_local_A)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_A)
              [-2], TRICKY_LDMATRIX_16x32_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(block_shared_local_B)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_B)
              [-2], TRICKY_LDMATRIX_16x32_B_TRANS_INTRIN)
sch.tensorize(kernel_i, TRICKY_MMA_i8i8i32_TRANS_INTRIN)

# sch.tensorize(sch.get_loops(block_local_C)[-2], MMA_store_16x16_f16_global_INTRIN)
write_sch(sch, log_path,
           "tensorize")

# unroll
# sch.unroll(init_block_b_i)
# sch.unroll(init_block_b_j)
# sch.unroll(block_shared_local_A_i)
# sch.unroll(block_shared_local_A_j)
# sch.unroll(block_shared_local_B_i)
# sch.unroll(block_shared_local_B_j)
# sch.unroll(ii)
# sch.unroll(jj)
# sch.unroll(A_shared_inner)
# sch.unroll(B_shared_inner)


write_sch(sch, log_path,
           "do_unroll")


ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# a_np = (np.ones(
#     (M // wmma_m, K // wmma_k, wmma_m, wmma_k)) * 2).astype("int8")
a_np = (np.random.rand
    (M // wmma_m, K // wmma_k, wmma_m, wmma_k) * 128).astype("int8")

b_np = (np.ones(
    (N // wmma_n, K // wmma_k, wmma_n, wmma_k))).astype("int8")
# b_np = (np.random.rand(
#     N // wmma_n, K // wmma_k, wmma_n, wmma_k) * 128).astype("int8")
cuda_a = tvm.nd.array((a_np).astype("int8"), ctx)
cuda_b = tvm.nd.array((b_np).astype("int8"), ctx)
cuda_c = tvm.nd.array(
    np.zeros((M // wmma_m, N // wmma_m, wmma_m, wmma_n)).astype("int32"), ctx)

if VERIFY:
    cuda_mod(cuda_a, cuda_b, cuda_c)
    a_np = a_np.transpose((0, 2, 1, 3)).reshape(M, N)
    b_np = b_np.transpose((0, 2, 1, 3)).reshape(N, K)
    c_np = cuda_c.numpy().transpose((0, 2, 1, 3)).reshape(M, N)
    np.testing.assert_allclose(
        c_np, np.matmul(a_np.astype("int32"), b_np.astype("int32").T), rtol=1e-4, atol=1e-4
    )

num_flops = 2 * M * K * N
num_runs = 1
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
