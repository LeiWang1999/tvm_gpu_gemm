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
    WMMA_FILL_16x16x16_F16_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
    WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
    WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN,
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
)


# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/tensorirscript_imma/" + fname
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
BM = 256
BN = 128
BK = 32
block_row_warps = 4
block_col_warps = 2
stage = 2
raster = 4
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [N, K], dtype="float16")
        C = T.match_buffer(c, [M, N], dtype="float16")

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("float16") * B[vj, vk].astype("float16")


ir_module = MyModule
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_b = sch.get_block("B")
block_shared = sch.cache_write(block_b, 0, "shared.dyn")

write_sch(sch, log_path, "cache_related")

(i, j, k) = sch.get_loops(block_b)
by, i = sch.split(i, factors=[None, BM])
bx, j = sch.split(j, factors=[None, BN])
bk, k = sch.split(k, factors=[None, BK])

write_sch(sch, log_path, "split_inner_loops")

bz, bx = sch.split(bx, factors=[None, raster])

sch.reorder(bz, by, bx, bk, i, j, k)
write_sch(sch, log_path, "reorder_inner_loops")

sch.bind(bx, "blockIdx.x")
sch.bind(by, "blockIdx.y")
sch.bind(bz, "blockIdx.z")

write_sch(sch, log_path, "block_bind")

block_b_tz, block_b_inner_i = sch.split(
    i, factors=[block_row_warps, None])

block_b_ty, block_b_inner_j = sch.split(
    j, factors=[block_col_warps, None])

sch.reorder(block_b_tz, block_b_ty, bk, k, block_b_inner_i, block_b_inner_j)

write_sch(sch, log_path, "split_outer_loops")

sch.bind(block_b_tz, "threadIdx.z")
sch.bind(block_b_ty, "threadIdx.y")

write_sch(sch, log_path, "thread_bind")

# schdule the shared memory

def fetch_to_shared(block, idx):
    block_read = sch.cache_read(block, idx, "shared.dyn")
    sch.compute_at(block_read, bk)
    vector_size = 8
    fused = sch.fuse(*sch.get_loops(block_read)[-2:])
    _, f_0, f_1, f_2, f_3 = sch.split(
        fused, factors=[None, block_row_warps, block_col_warps, warp_size, vector_size])
    sch.bind(f_2, "threadIdx.x")
    sch.bind(f_1, "threadIdx.y")
    sch.bind(f_0, "threadIdx.z")
    sch.vectorize(f_3)
    offset = 8
    sch.storage_align(block_read, 0, axis=-2, factor=32, offset=offset)

fetch_to_shared(block_b, 0)
fetch_to_shared(block_b, 1)
write_sch(sch, log_path, "shared_memory_schedule")


mma_m = 16
mma_n = 16
mma_k = 16

block_b_inner_i, block_b_inner_i_tc = sch.split(
    block_b_inner_i, factors=[None, mma_m])
block_b_inner_j, block_b_inner_j_tc = sch.split(
    block_b_inner_j, factors=[None, mma_n])
k, k_tc = sch.split(k, factors=[None, mma_k])

sch.reorder(k, block_b_inner_i, block_b_inner_j,
            block_b_inner_i_tc, block_b_inner_j_tc, k_tc)

write_sch(sch, log_path, "mma_tile")

A_warp = sch.cache_read(block_b, 0, "wmma.matrix_a")
B_warp = sch.cache_read(block_b, 1, "wmma.matrix_b")
sch.compute_at(A_warp, k)
sch.compute_at(B_warp, k)
C_warp = sch.cache_write(block_b, 0, "wmma.accumulator")
sch.reverse_compute_at(C_warp, block_b_ty)
write_sch(sch, log_path, "cache_read_write_warp")

ii, jj = sch.get_loops(C_warp)[-2:]
io, ii = sch.split(ii, factors=[None, mma_m])
jo, ji = sch.split(jj, factors=[None, mma_n])
sch.reorder(io, jo, ii, ji)


def tile_wmma_fragment(block_read, kernel_i, kernel_j):
    i, j = sch.get_loops(block_read)[-2:]
    i, kernel_i = sch.split(i, factors=[None, kernel_i])
    j, kernel_j = sch.split(j, factors=[None, kernel_j])
    sch.reorder(i, j, kernel_i, kernel_j)
    return kernel_i

loop_a = tile_wmma_fragment(A_warp, mma_m, mma_k)

loop_b = tile_wmma_fragment(B_warp, mma_n, mma_k)

write_sch(sch, log_path, "tile_fragment")


block_init_c = sch.decompose_reduction(
    block_b, bk)
write_sch(sch, log_path, "decompose_reduction")


sch.reverse_compute_at(block_shared, sch.get_loops(C_warp)[-3])
block_shared_i, block_shared_j = sch.get_loops(block_shared)[-2:]
fused = sch.fuse(block_shared_i, block_shared_j)
_, f_0, f_1 = sch.split(
        fused, factors=[None, warp_size, 8])
sch.bind(f_0, "threadIdx.x")
sch.vectorize(f_1)
write_sch(sch, log_path, "transform_layout")

sch.tensorize(loop_a, WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN)
sch.tensorize(loop_b, WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN)
write_sch(sch, log_path, "tensorize_ldmatrix")

sch.tensorize(block_b_inner_i_tc, WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)
write_sch(sch, log_path, "tensorize_mma_sync")

sch.tensorize(sch.get_loops(block_init_c)[-2], WMMA_FILL_16x16x16_F16_INTRIN)
write_sch(sch, log_path, "tensorize_mma_fill")

sch.tensorize(sch.get_loops(C_warp)[-2], WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN)
write_sch(sch, log_path, "tensorize_store")


if stage > 1:
    sch.annotate(bk, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 2, stage - 1, stage - 1])
    sch.annotate(bk, ann_key="software_pipeline_order", ann_val=[0, 1, 3, 2, 4])
    sch.annotate(bk, ann_key="software_pipeline_async_stages", ann_val=[0])

    # load local A, load local B, compute
    sch.annotate(k, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
    sch.annotate(k, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    code = code.replace("#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1", "#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0")
    return code

ctx = tvm.cuda(0)
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

a_np = (np.random.rand(M, K)).astype("float16")
b_np = (np.random.rand(K, N)).astype("float16")
cuda_a = tvm.nd.array((a_np).astype("float16"), ctx)
cuda_b = tvm.nd.array((b_np).astype("float16"), ctx)
cuda_c = tvm.nd.array(
    np.zeros((M, N)).astype("float16"), ctx)

if VERIFY:
    cuda_mod(cuda_a, cuda_b, cuda_c)
    c_np = cuda_c.numpy()
    np.testing.assert_allclose(
        c_np, np.matmul(a_np.astype("float16"), b_np.astype("float16").T), rtol=1e-1, atol=1e-1
    )

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
