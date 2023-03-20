import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from tvm.tir.tensor_intrin.cuda import (
    WMMA_FILL_16x16x16_S32_INTRIN,
    WMMA_LOAD_16x16x16_S8_A_INTRIN,
    WMMA_LOAD_16x16x16_S8_B_INTRIN,
    WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN,
    WMMA_SYNC_16x16x16_s8s8s32_INTRIN,
    WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN,
    WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN,
    WMMA_STORE_16x16x16_S32_SHARED_INTRIN
)

log_path = "progress/amos_with_tensorir_quantize_input_zero_point/0.wmma_int8_int32_nn"
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

warp_size = 32
block_row_warps = 2
block_col_warps = 4
warp_row_tiles = 8
warp_col_tiles = 1
chunk = 2
vec = 16
wmma_m = 16
wmma_n = 16
wmma_k = 16
split_k = 16

CONST_VAR = 12
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle, pb: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="int8")
        B = T.match_buffer(b, [K, N], dtype="int8")
        C = T.match_buffer(c, [M, N], dtype="int32")
        PB = T.match_buffer(pb, [N], dtype="int32")
        QC = T.alloc_buffer([M, N], dtype="int32", scope="shared")


        for i, j, k  in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    QC[vi, vj] = 0
                QC[vi, vj] = QC[vi, vj] + \
                    A[vi, vk].astype("int32") * B[vk, vj].astype("int32")
        
        for i, j in T.grid(M, N):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = QC[vi, vj] + CONST_VAR + PB[vj]


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(type(ir_module))
print(ir_module.script())

write_sch(sch, log_path, "original")

block_b = sch.get_block("B")
block_tricky_A = sch.cache_read(block_b, 0, "global")
block_tricky_shared_A = sch.cache_read(block_b, 0, "shared")
block_tricky_shared_local_A = sch.cache_read(block_b, 0, "wmma.matrix_a")
block_tricky_B = sch.cache_read(block_b, 1, "global")
block_tricky_shared_B = sch.cache_read(block_b, 1, "shared")
block_tricky_shared_local_B = sch.cache_read(block_b, 1, "wmma.matrix_b")
block_tricky_local_C = sch.cache_write(block_b, 0, "wmma.accumulator")
block_C = sch.get_block("C")
block_c_local = sch.cache_write(block_C, 0, "local")
write_sch(sch, log_path, "cache_related")
# sch.reverse_compute_inline(block_C)
write_sch(sch, log_path, "reverse_compute_inline")


def tricky_transform_A(i, j):
    return (i // wmma_m, j // wmma_k, i % wmma_m, j % wmma_k)


def tricky_transform_B(i, j):
    return (i // wmma_k, j // wmma_n, i % wmma_k, j % wmma_n)


def tricky_transform_C(i, j):
    return (i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n)

sch.transform_layout(block_tricky_A, ("write", 0),tricky_transform_A)
sch.transform_layout(block_tricky_B, ("write", 0),tricky_transform_B)
sch.transform_layout(block_tricky_shared_A, ("write", 0), tricky_transform_A)
sch.transform_layout(block_tricky_shared_B, ("write", 0), tricky_transform_B)
sch.transform_layout(block_tricky_shared_local_A, ("write", 0), tricky_transform_A)
sch.transform_layout(block_tricky_shared_local_B, ("write", 0), tricky_transform_B)
sch.transform_layout(block_b, ("write", 0), tricky_transform_C)
# sch.transform_layout(block_tricky_local_C, ("write", 0),tricky_transform_C)

write_sch(sch, log_path, "tricky_transform")


(i, j, k) = sch.get_loops(block_b)
i, kernel_i = sch.split(i, factors=[None, wmma_m])
j, kernel_j = sch.split(j, factors=[None, wmma_n])
k, kernel_k = sch.split(k, factors=[None, wmma_k])
sch.reorder(i, j, k, kernel_i, kernel_j, kernel_k)

write_sch(sch, log_path, "tricky_extract_compute")

block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)
block_k, block_j = sch.split(block_j, factors=[None, split_k])

write_sch(sch, log_path, "block_tile")

sch.bind(block_k, "blockIdx.z")
sch.bind(block_i, "blockIdx.y")
sch.bind(block_j, "blockIdx.x")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.z")

write_sch(sch, log_path, "thread_bind")


# cache read A from global memory to shared_memory
sch.compute_at(block_tricky_shared_local_A, ki)
sch.compute_at(block_tricky_shared_A, ko)
sch.compute_at(block_tricky_shared_local_B, ki)
sch.compute_at(block_tricky_shared_B, ko)
sch.reverse_compute_at(block_tricky_local_C, j)
sch.reverse_compute_at(block_C, j)
sch.reverse_compute_at(block_c_local, j)

write_sch(sch, log_path, "cache_read_compute_at")


def tricky_extract_cache(block, sub_i, sub_j):
    i, j = sch.get_loops(block)[-2:]
    i, kernel_i = sch.split(i, factors=[None, sub_i])
    j, kernel_j = sch.split(j, factors=[None, sub_j])
    sch.reorder(i, j, kernel_i, kernel_j)
    return (i, j, kernel_i, kernel_j)


block_tricky_shared_local_A_loops = tricky_extract_cache(
    block_tricky_shared_local_A, wmma_m, wmma_k)
block_tricky_shared_A_loops = tricky_extract_cache(
    block_tricky_shared_A, wmma_m, wmma_k)
block_tricky_shared_local_B_loops = tricky_extract_cache(
    block_tricky_shared_local_B, wmma_n, wmma_k)
block_tricky_shared_B_loops = tricky_extract_cache(
    block_tricky_shared_B, wmma_n, wmma_k)
block_tricky_local_C_loops = tricky_extract_cache(
    block_tricky_local_C, wmma_m, wmma_n)

write_sch(sch, log_path, "tricky_extract_cache")

sch.reverse_compute_at(block_C, block_tricky_local_C_loops[-3])
sch.reverse_compute_at(block_c_local, block_tricky_local_C_loops[-3])

write_sch(sch, log_path, "reverse_compute_at_cache_write")

block_C_i, block_C_j = sch.get_loops(block_C)[-2:]
block_C_fused_i_j = sch.fuse(block_C_i, block_C_j)
_, block_C_ty, block_C_tx = sch.split(block_C_fused_i_j, factors=[None, block_row_warps, warp_size])
sch.bind(block_C_ty, "threadIdx.y")
sch.bind(block_C_tx, "threadIdx.x")

write_sch(sch, log_path, "block_C_thread_bind")

block_c_local_i, block_c_local_j = sch.get_loops(block_c_local)[-2:]
block_c_local_fused_i_j = sch.fuse(block_c_local_i, block_c_local_j)
_, block_c_local_ty, block_c_local_tx = sch.split(block_c_local_fused_i_j, factors=[None, block_row_warps, warp_size])
sch.bind(block_c_local_ty, "threadIdx.y")
sch.bind(block_c_local_tx, "threadIdx.x")

write_sch(sch, log_path, "block_c_local_thread_bind")

# 128x32
A_shared_fused = sch.fuse(*sch.get_loops(block_tricky_shared_A)[-4:])
A_shared_ty, A_shared_tz, A_shared_inner, A_shared_tx, A_shared_vi = sch.split(
    A_shared_fused, factors=[block_row_warps, block_col_warps, None, warp_size, vec])
sch.vectorize(A_shared_vi)
sch.bind(A_shared_tx, "threadIdx.x")
sch.bind(A_shared_ty, "threadIdx.y")
sch.bind(A_shared_tz, "threadIdx.z")
write_sch(sch, log_path, "schedule_A_shared")

B_shared_fused = sch.fuse(*sch.get_loops(block_tricky_shared_B)[-4:])
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

sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_S32_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
sch.tensorize(sch.get_loops(block_tricky_shared_local_A)[-2], WMMA_LOAD_16x16x16_S8_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
sch.tensorize(sch.get_loops(block_tricky_shared_local_B)[-2], WMMA_LOAD_16x16x16_S8_B_INTRIN)
sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_s8s8s32_INTRIN)
sch.tensorize(sch.get_loops(block_tricky_local_C)
              [-2], WMMA_STORE_16x16x16_S32_SHARED_INTRIN)
write_sch(sch, log_path,
           "tensorize")




# sch.reverse_compute_at(block_C, j)
# sch.reverse_compute_at(block_c_local, j)

# unroll
write_sch(sch, log_path,
           "do_unroll")

def schedule_tricky_transform(block, vec):
    i, j = sch.get_loops(block)[-2:]
    if K <= 16384:
        fused_axis = sch.fuse(i, j)
        # 16384
        by, bx, vx, ty, tx, fused_inner, fused_vi = sch.split(
            fused_axis, factors=[4, 2048, 1, 128, 8, None, vec])
        # 8192
        # by, bx, vx, ty, tx, fused_inner, fused_vi = sch.split(
        #     fused_axis, factors=[256, 256, 4, 2, 8, None, vec])
        
        sch.vectorize(fused_vi)
        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        # sch.unroll(fused_inner)
    else:
        bx, fused_inner, ty, tx, fused_vi = sch.split(
            j, factors=[1024, None, 32, 32, vec])
        sch.vectorize(fused_vi)
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

schedule_tricky_transform(block_tricky_A, vec=vec)
schedule_tricky_transform(block_tricky_B, vec=vec)
# schedule_tricky_transform(block_tricky_C, vec=4)

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype("int8"), ctx)
cuda_b = tvm.nd.array(np.arange(N * K).reshape((K, N)).astype("int8"), ctx)
cuda_c = tvm.nd.array(np.zeros((M, N)).astype("int32"), ctx)
cuda_d = tvm.nd.array(np.zeros(M).astype("int32"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c, cuda_d)

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c, cuda_d).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
