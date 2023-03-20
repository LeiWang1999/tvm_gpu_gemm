"""
    Considering a gemm problem, in this part we try to leverage the ldmatrix, mma, and stmatrix to do the computation.
    The ldmatrix and stmatrix are used to load and store the data from global memory to shared memory.
    The mma is used to do the computation.
    thread_x will be set into 32, which represents the number of threads in a warp.
    thread_y and thread_z will be set into value which represents the array of warps. 
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


log_path = "progress/amos_with_tensorir_quantize_4terms/5.padding_mma_i32_nt"
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

BM = 128
BN = 256
BK = 64
warp_size = 32
block_row_warps = 2
block_col_warps = 4

Z0 = 1
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="int8")
        B = T.match_buffer(b, [N, K], dtype="int8")
        C = T.match_buffer(c, [M, N], dtype="int32")
        PA = T.alloc_buffer([M], dtype="int32", scope="shared")

        for i, k in T.grid(M, K):
            with T.block("Pre_compute_A"):
                vi, vk = T.axis.remap("SR", [i, k]) 
                with T.init():
                    PA[vi] = 0
                PA[vi] = PA[vi] + Z0 * A[vi, vk].astype("int32")

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

block_pre_compute_a = sch.get_block("Pre_compute_A")
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

block_b_tz, block_b_inner_i = sch.split(
    i, factors=[block_row_warps, None])

block_b_ty, block_b_inner_j = sch.split(
    j, factors=[block_col_warps, None])

sch.reorder(block_b_tz, block_b_ty, bk, block_b_inner_i, block_b_inner_j, k)

write_sch(sch, log_path, "split_outer_loops")

sch.bind(block_b_tz, "threadIdx.z")
sch.bind(block_b_ty, "threadIdx.y")

write_sch(sch, log_path, "thread_bind")

def get_extent(loop):
    return sch.get_sref(loop).stmt.extent
sch.reverse_compute_at(block_pre_compute_a, bk)
pa_i, reduce_k = sch.get_loops(block_pre_compute_a)[-2:]
by, pa_i = sch.split(pa_i, factors=[get_extent(by), None])
sch.bind(by, "blockIdx.y")
# schdule the shared memory

def fetch_to_shared(block, idx):
    block_read = sch.cache_read(block, idx, "shared")
    sch.compute_at(block_read, bk)
    vector_size = 16
    fused = sch.fuse(*sch.get_loops(block_read)[-2:])
    _, f_1, f_2, f_3 = sch.split(
        fused, factors=[None, block_col_warps, warp_size, vector_size])
    sch.bind(f_2, "threadIdx.x")
    sch.bind(f_1, "threadIdx.y")
    sch.vectorize(f_3)
    offset = 0
    sch.storage_align(block_read, 0, axis=-2, factor=32, offset=offset)

# schedule A
fetch_to_shared(block_b, 0)
# schedule B
fetch_to_shared(block_b, 1)
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

sch.reorder(block_b_inner_i, block_b_inner_j,
            k, block_b_inner_i_tc, block_b_inner_j_tc, k_tc)

write_sch(sch, log_path, "mma_tile")

# block_inner = sch.blockize(block_b_inner_i_tc)
# block_outer, block_inner = block_inner, block_b
write_sch(sch, log_path, "blockize")

A_warp = sch.cache_read(block_b, 0, "warp", consumer_blocks=[block_b])
B_warp = sch.cache_read(block_b, 1, "warp", consumer_blocks=[block_b])
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
    return i

loop_a = tile_wmma_fragment(A_warp, mma_m, mma_k)

loop_b = tile_wmma_fragment(B_warp, mma_n, mma_k)

write_sch(sch, log_path, "tile_fragment")


block_init_c = sch.decompose_reduction(
    block_b, bk)
write_sch(sch, log_path, "decompose_reduction")

def index_map_A(i, j):
    return (
        i // 16,
        j // 32,
        *shared_16x32_to_ldmatrix_32x16_layout(i % 16, j % 32),
    )

def index_map_B(i, j):
    return (
        i // 32,
        j // 16,
        *shared_32x16_to_ldmatrix_32x16_layout(i % 32, j % 16),
    )

def index_map_C(i, j):
    return (
        i // 16,
        j // 16,
        *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
    )


sch.transform_layout(A_warp, ("write", 0), index_map_A)
sch.transform_layout(B_warp, ("write", 0), index_map_A)
sch.transform_layout(C_warp, ("read", 0), index_map_C)

write_sch(sch, log_path, "transform_layout")

sch.tensorize(loop_a, LDMATRIX_16x32_A_INTRIN)
sch.tensorize(loop_b, LDMATRIX_16x32_B_TRANS_INTRIN)
write_sch(sch, log_path, "tensorize_ldmatrix")

sch.tensorize(block_b_inner_i_tc, MMA_i8i8i32_TRANS_INTRIN)

sch.tensorize(sch.get_loops(block_init_c)[-2], MMA_fill_16x16_i32_INTRIN)
sch.tensorize(sch.get_loops(C_warp)[-2], MMA_store_16x16_i32_global_INTRIN)

write_sch(sch, log_path, "tensorize")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")
a_np = (np.random.uniform(
    size=(M, K)) * 128).astype("int8")
b_np = (np.random.uniform(
    size=(K, N)) * 128).astype("int8")
cuda_a = tvm.nd.array((a_np).astype("int8"), ctx)
cuda_b = tvm.nd.array((b_np).astype("int8"), ctx)
cuda_c = tvm.nd.array(
    np.zeros((M, N)).astype("int32"), ctx)

if VERIFY:
    cuda_mod(cuda_a, cuda_b, cuda_c)
    c_np = cuda_c.numpy()
    np.testing.assert_allclose(
        c_np, np.matmul(a_np.astype("int32"), b_np.astype("int32").T), rtol=1e-1, atol=1e-1
    )

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
