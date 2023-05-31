import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
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


VERIFY = True

M = 256
N = 8192
K = 8192
if VERIFY:
    M = 256
    N = 4096
    K = 1024

warp_size = 32
wmma_m = 16
wmma_n = 16
wmma_k = 16

BM = 128
BN = 256
BK = 32
block_row_warps = 2
block_col_warps = 2
warp_row_tiles = BM // (wmma_m * block_row_warps)  
warp_col_tiles = BN // (wmma_n * block_col_warps)
chunk = BK // (wmma_k)
raster = 16
stage = 1
shared_pad = 8
vec = 8

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
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(type(ir_module))
print(ir_module.script())

write_sch(sch, log_path, "original")
block_b = sch.get_block("B")
block_shared_A = sch.cache_read(block_b, 0, "shared")
block_shared_local_A = sch.cache_read(block_b, 0, "wmma.matrix_a")
block_shared_B = sch.cache_read(block_b, 1, "shared")
block_shared_local_B = sch.cache_read(block_b, 1, "wmma.matrix_b")
block_local_C = sch.cache_write(block_b, 0, "wmma.accumulator")

write_sch(sch, log_path, "cache_related")

(i, j, k) = sch.get_loops(block_b)
i, kernel_i = sch.split(i, factors=[None, wmma_m])
j, kernel_j = sch.split(j, factors=[None, wmma_n])
k, kernel_k = sch.split(k, factors=[None, wmma_k])
block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
block_j, block_k = sch.split(block_j, factors=[None, raster])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_k, block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)

write_sch(sch, log_path, "block_tile")
sch.bind(block_k, "blockIdx.z")
sch.bind(block_i, "blockIdx.y")
sch.bind(block_j, "blockIdx.x")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.z")

write_sch(sch, log_path, "thread_bind")


# cache read A from global memory to shared_memory
sch.compute_at(block_shared_local_A, ki)
sch.compute_at(block_shared_A, ko, preserve_unit_loops=True)
sch.compute_at(block_shared_local_B, ki)
sch.compute_at(block_shared_B, ko, preserve_unit_loops=True)
sch.reverse_compute_at(block_local_C, j)
write_sch(sch, log_path, "cache_read_compute_at")


A_shared_fused = sch.fuse(*sch.get_loops(block_shared_A)[-2:])
A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
    A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
sch.vectorize(A_shared_vi)
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
sch.bind(A_shared_tz, "threadIdx.z")
sch.storage_align(block_shared_A, 0, axis=-2, factor=32, offset=shared_pad)
write_sch(sch, log_path, "schedule_A_shared")

B_shared_fused = sch.fuse(*sch.get_loops(block_shared_B)[-2:])
B_shared_ty, B_shared_tz, B_shared_inner, B_shared_tx, B_shared_vi = sch.split(
    B_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
sch.vectorize(B_shared_vi)
sch.bind(B_shared_tx, "threadIdx.x")
sch.bind(B_shared_ty, "threadIdx.y")
sch.bind(B_shared_tz, "threadIdx.z")
sch.storage_align(block_shared_B, 0, axis=-2, factor=32, offset=shared_pad)
write_sch(sch, log_path, "schedule_B_shared")

A_local_i, A_local_j = sch.get_loops(block_shared_local_A)[-2:]
A_local_i, A_local_kernel_i = sch.split(A_local_i, factors=[None, wmma_m])
A_local_j, A_local_kernel_j = sch.split(A_local_j, factors=[None, wmma_k])
sch.reorder(A_local_i, A_local_j, A_local_kernel_i, A_local_kernel_j)

B_local_i, B_local_j = sch.get_loops(block_shared_local_B)[-2:]
B_local_i, B_local_kernel_i = sch.split(B_local_i, factors=[None, wmma_n])
B_local_j, B_local_kernel_j = sch.split(B_local_j, factors=[None, wmma_k])
sch.reorder(B_local_i, B_local_j, B_local_kernel_i, B_local_kernel_j)

C_local_i, C_local_j = sch.get_loops(block_local_C)[-2:]
C_local_i, C_local_kernel_i = sch.split(C_local_i, factors=[None, wmma_m])
C_local_j, C_local_kernel_j = sch.split(C_local_j, factors=[None, wmma_n])
sch.reorder(C_local_i, C_local_j, C_local_kernel_i, C_local_kernel_j)

# decompose reduction
init_block_b = sch.decompose_reduction(block_b, ko)
write_sch(sch, log_path, "decompose_reduction")

# transpose layout

init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_F16_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(block_shared_local_A)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_A)
              [-2], WMMA_LOAD_16x16x16_F16_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(block_shared_local_B)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_B)
              [-2], WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN)
sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)

sch.tensorize(sch.get_loops(block_local_C)[-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)
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

if stage > 1:

    sch.annotate(ki, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
    sch.annotate(ki, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

    sch.annotate(ko, ann_key="software_pipeline_stage",
                 ann_val=[0, 0, 0, stage - 1, 0])
    sch.annotate(ko, ann_key="software_pipeline_order",
                 ann_val=[0, 1, 3, 2, 4])
    sch.annotate(ko, ann_key="software_pipeline_async_stages", ann_val=[0])
    
write_sch(sch, log_path,
           "do_unroll")


ctx = tvm.cuda(0)
with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
    cuda_mod = tvm.build(sch.mod, target="cuda")
    
write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

a_np = (np.random.rand(M, K)).astype("float16")
b_np = (np.random.rand(N, K)).astype("float16")
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


# cuda_mod(cuda_a, cuda_b, cuda_c)
# print(cuda_c.numpy())
num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
