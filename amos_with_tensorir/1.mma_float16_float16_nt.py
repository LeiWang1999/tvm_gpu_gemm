import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from intrin.tricky_mma_float16_float16 import (
    TRICKY_MMA_A_G2S_16x16_f16_INTRIN,
    TRICKY_MMA_B_G2S_16x16_f16_INTRIN,
    TRICKY_MMA_B_TRANS_G2S_16x16_f16_INTRIN,
    TRICKY_MMA_fill_16x16_f16_INTRIN,
    TRICKY_LDMATRIX_16x16_A_INTRIN,
    TRICKY_LDMATRIX_16x16_B_INTRIN,
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
    TRICKY_MMA_f16f16f16_INTRIN,
    TRICKY_MMA_f16f16f16_TRANS_INTRIN,
    TRICKY_MMA_store_16x16_f16_global_INTRIN,
    A_global_16x16_to_shared_load_16x16_layout,
    B_global_16x16_to_shared_load_16x16_layout,
    C_shared_16x16_to_ldmatrix_32x8_layout,
    A_B_shared_16x16_to_ldmatrix_32x8_layout
)

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/amos_with_tensorir/" + fname
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
    N = 4096
    K = 1024

warp_size = 32
block_row_warps = 2
block_col_warps = 2
warp_row_tiles = 4
warp_col_tiles = 8
# block_row_warps = 4
# block_col_warps = 2
# warp_row_tiles = 4
# warp_col_tiles = 2
chunk = 2
vec = 8
wmma_m = 16
wmma_n = 16
wmma_k = 16
splitk = 16


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
                    C[vi, vj] = T.float16(0.0)
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("float16") * B[vj, vk].astype("float16")


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

print(type(ir_module))
print(ir_module.script())

write_sch(sch, log_path, "original")

block_b = sch.get_block("B")
block_tricky_A = sch.cache_read(block_b, 0, "global")
block_tricky_shared_A = sch.cache_read(block_b, 0, "shared")
block_tricky_shared_local_A = sch.cache_read(block_b, 0, "warp")
block_tricky_B = sch.cache_read(block_b, 1, "global")
block_tricky_shared_B = sch.cache_read(block_b, 1, "shared")
block_tricky_shared_local_B = sch.cache_read(block_b, 1, "warp")
# block_tricky_C = sch.cache_write(block_b, 0, "global")
block_tricky_local_C = sch.cache_write(block_b, 0, "warp")

write_sch(sch, log_path, "cache_related")


def tricky_transform_A(i, j):
    return (i // wmma_m, j // wmma_k, i % wmma_m, j % wmma_k)


def tricky_transform_B(i, j):
    return (i // wmma_n, j // wmma_k, i % wmma_n, j % wmma_k)


def tricky_transform_C(i, j):
    return (i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n)


sch.transform_layout(block_tricky_A, ("write", 0), tricky_transform_A)
sch.transform_layout(block_tricky_B, ("write", 0), tricky_transform_B)
sch.transform_layout(block_tricky_shared_A, ("write", 0), tricky_transform_A)
sch.transform_layout(block_tricky_shared_B, ("write", 0), tricky_transform_B)
sch.transform_layout(block_tricky_shared_local_A,
                     ("write", 0), tricky_transform_A)
sch.transform_layout(block_tricky_shared_local_B,
                     ("write", 0), tricky_transform_B)
sch.transform_layout(block_b, ("write", 0), tricky_transform_C)
# sch.transform_layout(block_tricky_local_C, ("write", 0), tricky_transform_C)

write_sch(sch, log_path, "tricky_transform_kernel")

(i, j, k) = sch.get_loops(block_b)
i, kernel_i = sch.split(i, factors=[None, wmma_m])
j, kernel_j = sch.split(j, factors=[None, wmma_n])
k, kernel_k = sch.split(k, factors=[None, wmma_k])
sch.reorder(i, j, k, kernel_i, kernel_j, kernel_k)
write_sch(sch, log_path, "tricky_extract")

block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
ko, ki = sch.split(k, factors=[None, chunk])
sch.reorder(block_i, block_j, i, j, ko, ki, ii,
            jj, kernel_i, kernel_j, kernel_k)

block_k, block_j = sch.split(block_j, factors=[None, splitk])
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


# 128x32
def permutation_A(i, j, kernel_i, kernel_j):
    return (i, j, *A_global_16x16_to_shared_load_16x16_layout(kernel_i, kernel_j))

# 128x32
def permutation_B(i, j, kernel_i, kernel_j):
    return (i, j, *B_global_16x16_to_shared_load_16x16_layout(kernel_i, kernel_j))

sch.transform_layout(block_tricky_shared_A, ("read", 0),
                     permutation_A, rewrite_type=1)
sch.transform_layout(block_tricky_shared_B, ("read", 0),
                     permutation_B, rewrite_type=1)
# sch.tensorize(sch.get_loops(block_tricky_shared_A)[-2], TRICKY_MMA_A_G2S_16x16_f16_INTRIN)
# block_tricky_shared_A = sch.get_block("A_g2s_shared")
# sch.tensorize(sch.get_loops(block_tricky_shared_B)[-2], TRICKY_MMA_B_TRANS_G2S_16x16_f16_INTRIN)
# block_tricky_shared_B = sch.get_block("B_g2s_shared_trans")


write_sch(sch, log_path, "tricky_shared_transform_layout")

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

# transpose layout


def index_map_A(i, k, wmma_m, wmma_k):
    return (i, k, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_k))


def index_map_B(j, k, wmma_n, wmma_k):
    return (j, k, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_n, wmma_k),)


def index_map_C(i, j, wmma_m, wmma_n):
    return (i, j, *C_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_n),)


sch.transform_layout(block_tricky_shared_local_A, ("write", 0), index_map_A)
sch.transform_layout(block_tricky_shared_local_B, ("write", 0), index_map_A)
sch.transform_layout(block_tricky_local_C, ("read", 0), index_map_C)
write_sch(sch, log_path, "transform_layout")

init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
sch.tensorize(sch.get_loops(init_block_b)
              [-2], TRICKY_MMA_fill_16x16_f16_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
block_tricky_shared_local_A_i, block_tricky_shared_local_A_j = sch.get_loops(
    block_tricky_shared_local_A)[-4:-2]
sch.tensorize(sch.get_loops(block_tricky_shared_local_A)
              [-2], TRICKY_LDMATRIX_16x16_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
block_tricky_shared_local_B_i, block_tricky_shared_local_B_j = sch.get_loops(
    block_tricky_shared_local_B)[-4:-2]
sch.tensorize(sch.get_loops(block_tricky_shared_local_B)
              [-2], TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN)
sch.tensorize(kernel_i, TRICKY_MMA_f16f16f16_TRANS_INTRIN)

sch.tensorize(sch.get_loops(block_tricky_local_C)
              [-2], TRICKY_MMA_store_16x16_f16_global_INTRIN)
write_sch(sch, log_path,
          "tensorize")

# schdule tricky transform

def schedule_tricky_transform(block, vec):
    i, j = sch.get_loops(block)[-2:]
    if K <= 16384:
        fused_axis = sch.fuse(i, j)
        # 16384
        by, bx, vx, ty, tx, fused_inner, fused_vi = sch.split(
            fused_axis, factors=[8192, 32, 1, 1, 8, None, vec])
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
# schedule_tricky_transform(block_tricky_C, vec=2)

# sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
# sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

# sch.annotate(k1, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
# sch.annotate(k1, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

# c_warp_o = sch.get_block("C_warp_o")
# print(sch.get_loops(c_warp_o)[-1])
# _, vec = sch.split(sch.get_loops(c_warp_o)[-1], factors=[None, 2])
# sch.vectorize(vec)

# unroll
# sch.unroll(init_block_b_i)
# sch.unroll(init_block_b_j)
# sch.unroll(block_tricky_shared_local_A_i)
# sch.unroll(block_tricky_shared_local_A_j)
# sch.unroll(block_tricky_shared_local_B_i)
# sch.unroll(block_tricky_shared_local_B_j)
# sch.unroll(ii)
# sch.unroll(jj)
# sch.unroll(A_shared_inner)
# sch.unroll(B_shared_inner)


write_sch(sch, log_path,
          "do_unroll")


ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

a_np = (np.random.rand(
    M, K) * 4).astype("float16")
b_np = (np.random.rand(
    N, K) * 4).astype("float16")

cuda_a = tvm.nd.array((a_np).astype("float16"), ctx)
cuda_b = tvm.nd.array((b_np).astype("float16"), ctx)
cuda_c = tvm.nd.array(
    np.zeros((M, N)).astype("float16"), ctx)

if VERIFY:
    cuda_mod(cuda_a, cuda_b, cuda_c)
    c_np = cuda_c.numpy()
    import torch
    a_torch = torch.tensor(a_np, device="cuda")
    b_torch = torch.tensor(b_np, device="cuda")
    c_torch = torch.tensor(c_np, device="cuda")
    torch.matmul(a_torch, b_torch.T, out=c_torch)
    c_torch_np = c_torch.cpu().numpy()
    print("torch result: ", c_torch_np[0][0:10])
    print("tvm result: ", c_np[0][0:10])
    np.testing.assert_allclose(
        c_np, c_torch_np, rtol=1e0, atol=1e0
    )

num_flops = 2 * M * K * N
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
