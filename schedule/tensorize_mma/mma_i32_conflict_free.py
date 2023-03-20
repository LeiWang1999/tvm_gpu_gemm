"""
    Considering a gemm problem, in this part we try to leverage the ldmatrix, mma, and stmatrix to do the computation.
    The ldmatrix and stmatrix are used to load and store the data from global memory to shared memory.
    The mma is used to do the computation.
    thread_x will be set into 32, which represents the number of threads in a warp.
    thread_y and thread_z will be set into value which represents the array of warps. 
    To archieve sota performance, we need to permutate the data in shared memory and the data in warp.
"""
import tvm
from tvm.script import tir as T
import numpy as np
import os
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_16x16_A_INTRIN,
    LDMATRIX_16x16_B_INTRIN,
    LDMATRIX_16x16_B_TRANS_INTRIN,
    LDMATRIX_16x32_A_INTRIN,
    LDMATRIX_32x16_B_INTRIN,
    LDMATRIX_16x32_B_TRANS_INTRIN,
    MMA_f16f16f32_INTRIN,
    MMA_f16f16f32_TRANS_INTRIN,
    MMA_f16f16f16_INTRIN,
    MMA_f16f16f16_TRANS_INTRIN,
    MMA_i8i8i32_INTRIN,
    MMA_i8i8i32_TRANS_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_fill_16x16_f16_INTRIN,
    MMA_fill_16x16_i32_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    MMA_store_16x16_f16_global_INTRIN,
    MMA_store_16x16_i32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
    shared_32x16_to_ldmatrix_32x16_layout,
    shared_16x32_to_ldmatrix_32x16_layout,
)


log_path = "progress/tensorize_mma/mma_i32_conflict_free"
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
BM = 128
BN = 256
BK = 64
warp_size = 32
block_row_warps = 2
block_col_warps = 4

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="int8")
        B = T.match_buffer(b, [N, K], dtype="int8")
        C = T.match_buffer(c, [M, N], dtype="int32")

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.int32(0)
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("int32") * B[vj, vk].astype("int32")


ir_module = MyModule
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_b = sch.get_block("B")
# C_wrap = sch.cache_write(block_b, 0, "local")
write_sch(sch, log_path, "cache_related")

(i, j, k) = sch.get_loops(block_b)
by, i = sch.split(i, factors=[None, BM])
bx, j = sch.split(j, factors=[None, BN])
bk, k = sch.split(k, factors=[None, BK])

write_sch(sch, log_path, "split_inner_loops")

sch.reorder(by, bx, bk, i, j, k)
write_sch(sch, log_path, "reorder_inner_loops")

sch.bind(bx, "blockIdx.x")
sch.bind(by, "blockIdx.y")

write_sch(sch, log_path, "block_bind")

# currently, we have a sub-problem which size is 128x256x64, to do the computation, we need to use 2 warps in row and 4 warps in column, so the value of thread_x, thread_y, thread_z will be 32, 4, 2, respectively, so the block_row_warps will be 4, the block_col_warps will be 2.

# because each warp will produce a 64x64 matrix, but the size of mma is 16x16, so we need to use 4x4 = 16 mma to do the computation, so the value of warp_row_tiles and warp_col_tiles will be 4, 4, or 2, 8, respectively.

# i->128, block_row_warps->2, result: 64
block_b_tz, block_b_inner_i = sch.split(
    i, factors=[block_row_warps, None])

# j->256, block_col_warps->4, result: 64
block_b_ty, block_b_inner_j = sch.split(
    j, factors=[block_col_warps, None])
# k->64
sch.reorder(block_b_tz, block_b_ty, bk, block_b_inner_i, block_b_inner_j, k)

write_sch(sch, log_path, "split_outer_loops")

sch.bind(block_b_tz, "threadIdx.z")
sch.bind(block_b_ty, "threadIdx.y")

write_sch(sch, log_path, "thread_bind")

# schdule the shared memory

def fetch_to_shared(block, idx):
    block_read = sch.cache_read(block, idx, "shared")

    def shared_conflict_free_permutation(i, j):
        def shared_8x64_to_free_4x128_layout(i, j):
            elements_per_thread = 16
            element_id = (i << 6) + j
            lane_id = element_id // elements_per_thread
            element_id_in_thread = element_id % elements_per_thread
            shared_c = lane_id % 8
            shared_s = lane_id // 8
            shared_row = (shared_c & 1) | ((shared_c >> 1) & 2)
            shared_col = ((shared_c << 1) & 4) | shared_s ^ shared_row
            return (shared_row, shared_col * elements_per_thread + element_id_in_thread)
        return (i // 8, j // 64, *shared_8x64_to_free_4x128_layout(i % 8, j % 64))

    sch.transform_layout(block_read, ("write", 0), shared_conflict_free_permutation)
    sch.compute_at(block_read, bk)
    vector_size = 16
    fused = sch.fuse(*sch.get_loops(block_read)[-2:])
    _, f_0, f_1, f_2, f_3 = sch.split(
        fused, factors=[None, block_row_warps, block_col_warps, warp_size, vector_size])
    sch.bind(f_2, "threadIdx.x")
    sch.bind(f_1, "threadIdx.y")
    sch.bind(f_0, "threadIdx.z")
    sch.vectorize(f_3)
    return block_read

# schedule A
block_shared_A = fetch_to_shared(block_b, 0)
# schedule B
block_shared_B = fetch_to_shared(block_b, 1)

write_sch(sch, log_path, "shared_memory_schedule")

# blockize for mma tensorize

mma_m = 16
mma_n = 16
mma_k = 32

block_b_inner_i, block_b_inner_i_tc = sch.split(
    block_b_inner_i, factors=[None, mma_m])
block_b_inner_j, block_b_inner_j_tc = sch.split(
    block_b_inner_j, factors=[None, mma_n])
k, k_tc = sch.split(k, factors=[None, mma_k])

sch.reorder(k, block_b_inner_i, block_b_inner_j, 
        block_b_inner_i_tc, block_b_inner_j_tc, k_tc)

write_sch(sch, log_path, "mma_tile")

A_warp = sch.cache_read(block_b, 0, "warp")
B_warp = sch.cache_read(block_b, 1, "warp")
sch.compute_at(A_warp, k)
sch.compute_at(B_warp, k)
C_warp = sch.cache_write(block_b, 0, "warp")
sch.reverse_compute_at(C_warp, block_b_ty)
write_sch(sch, log_path, "cache_read_write_warp")

ii, jj = sch.get_loops(C_warp)[-2:]
io, ii = sch.split(ii, factors=[None, mma_m])
jo, ji = sch.split(jj, factors=[None, mma_n])
sch.reorder(io, jo, ii, ji)


def tile_wmma_fragment(block_read, height, width):
    i, j = sch.get_loops(block_read)[-2:]
    i0, i1 = sch.split(i, factors=[None, height])
    j0, j1 = sch.split(j, factors=[None, width])
    sch.reorder(i0, j0, i1, j1)
    return i1

# ldmatrix.sync.aligned.x4.m8n8.shared.b16 read a 32x16 matrix
loop_a = tile_wmma_fragment(A_warp, 32, 16)
loop_b = tile_wmma_fragment(B_warp, 32, 16)

write_sch(sch, log_path, "tile_fragment")

block_init_c = sch.decompose_reduction(
    block_b, bk)
write_sch(sch, log_path, "decompose_reduction")

# transform layout for A, B warp
write_sch(sch, log_path, "transform_layout")

# def shared_32x16_to_ldmatrix_32x16(m, n):
#     pass
# raw size 64x16
def warp_write_permutation(m, n):
    return (m // 32, n // 16, *shared_32x16_to_ldmatrix_32x16_layout(m % 32, n % 16))


sch.transform_layout(A_warp, ("write", 0), warp_write_permutation)
sch.transform_layout(B_warp, ("write", 0), warp_write_permutation)

def warp_read_permutation(m, n):
    return (
        m // 16,
        n // 16,
        *shared_16x16_to_ldmatrix_32x8_layout(m % 16, n % 16),
    )


sch.transform_layout(C_warp, ("read", 0), warp_read_permutation)

write_sch(sch, log_path, "warp_permutation")


# tensorize ldmatrix
# tensorize mma.m8n8k16
# tensorize fill_fragment
# tensorize stmatrix
sch.tensorize(loop_a, LDMATRIX_32x16_B_INTRIN)
sch.tensorize(loop_b, LDMATRIX_32x16_B_INTRIN)

write_sch(sch, log_path, "tensorize_ldmatrix")

# _test_block = sch.get_block("")
sch.tensorize(block_b_inner_i_tc, MMA_i8i8i32_TRANS_INTRIN)

sch.tensorize(sch.get_loops(block_init_c)[-2], MMA_fill_16x16_i32_INTRIN)
sch.tensorize(sch.get_loops(C_warp)[-2], MMA_store_16x16_i32_global_INTRIN)

write_sch(sch, log_path, "tensorize")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype("int8"), ctx)
cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype("int8"), ctx)
cuda_c = tvm.nd.array(np.zeros((M, N)).astype("int32"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
