import tvm
from tvm import te
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_16x16_A_INTRIN,
    LDMATRIX_16x16_B_INTRIN,
    MMA_f16f16f16_INTRIN,
    MMA_fill_16x16_f16_INTRIN,
    MMA_store_16x16_f16_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
)
# import tvm convert
from tvm.runtime import convert

log_path = "progress/tensorscript_imma/4.tricky_mma_float16_float16_nn"
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


def get_ldmatrix_intrin(dtype, is_b, transposed, shared_scope="shared"):
    m_dim = 16
    k_dim = 16
    warp_size = 32
    half_warp = warp_size // 2
    half_warp_expr = convert(half_warp)
    local_size = (m_dim * k_dim) // warp_size
    shared_offset = None
    index_map = None

    if transposed:
        assert False, "Transposed matrix not supported"

    ldmatrix_col_major = is_b and not transposed

    if k_dim == 16:
        assert dtype == "float16"

        index_map = shared_16x16_to_ldmatrix_32x8_layout

        def shared_offset(tx, stride): return stride * (tx % half_warp_expr) + 8 * (
            tx // half_warp_expr
        )
    else:
        assert False, "Only k_dim == 16 (float16) or k_dim == 32 (int8) supported for now"

    assert index_map and shared_offset

    if is_b and not transposed:
        row_dim = k_dim
        col_dim = m_dim
    else:
        row_dim = m_dim
        col_dim = k_dim

    shmem_shape = (row_dim, col_dim)

    @T.prim_func
    def ldmatrix_desc(warp_handle: T.handle, shared_handle: T.handle) -> None:
        shared = T.match_buffer(
            shared_handle,
            shmem_shape,
            dtype,
            align=64,
            offset_factor=16,
            scope=shared_scope,
        )
        warp = T.match_buffer(
            warp_handle, (warp_size, local_size), dtype, align=64, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(shared[0:row_dim, 0:col_dim])
            T.writes(warp[0:warp_size, 0:local_size])

            for ax0, ax1 in T.grid(row_dim, col_dim):
                with T.block("shared_warp"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(shared[v0, v1])

                    thread_id, local_id = T.meta_var(index_map(v0, v1))
                    T.writes(warp[thread_id, local_id])
                    warp[thread_id, local_id] = shared[v0, v1]

    @T.prim_func
    def ldmatrix_impl(warp_handle: T.handle, shared_handle: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        shared = T.match_buffer(
            shared_handle,
            shmem_shape,
            dtype,
            align=64,
            offset_factor=16,
            scope=shared_scope,
            strides=[s0, s1],
        )
        warp = T.match_buffer(
            warp_handle, (warp_size, local_size), dtype, align=64, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(shared[0:row_dim, 0:col_dim])
            T.writes(warp[0:warp_size, 0:local_size])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, warp_size)

            T.evaluate(
                T.ptx_ldmatrix(
                    ldmatrix_col_major,
                    4,  # Always load 4 matrices
                    ".b16",
                    warp.data,
                    warp.elem_offset + convert(local_size) * tx,
                    shared.access_ptr("r"),
                    shared_offset(tx, s0),
                    dtype=dtype,
                )
            )

    return ldmatrix_desc, ldmatrix_impl


M = 16384
N = 16384
K = 16384
warp_size = 32
block_row_warps = 4
block_col_warps = 2
warp_row_tiles = 2
warp_col_tiles = 4
chunk = 2
vec = 8
wmma_m = 16
wmma_n = 16
wmma_k = 16

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M // wmma_m, K // wmma_k, wmma_m, wmma_k], dtype="float16")
        B = T.match_buffer(b, [K // wmma_k, N // wmma_n, wmma_k, wmma_n], dtype="float16")
        C = T.match_buffer(c, [M // wmma_m, N // wmma_n,
                           wmma_m, wmma_n], dtype="float16")

        for ii, jj, kk, i, j, k  in T.grid(M // wmma_m, N // wmma_n, K // wmma_k, wmma_m, wmma_n, wmma_k):
            with T.block("B"):
                vii, vjj, vkk, vi, vj, vk = T.axis.remap("SSRSSR", [ii, jj, kk, i, j, k])
                with T.init():
                    C[vii, vjj, vi, vj] = 0.0
                C[vii, vjj, vi, vj] = C[vii, vjj, vi, vj] + \
                    A[vii, vkk, vi, vk] * B[vkk, vjj, vk, vj]


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

# transpose layout


def _shared_16x16_to_ldmatrix_32x8_layout(i, j):
    return (i * 2 + j // 8, j % 8)

def index_map_B(i, j, wmma_i, wmma_j):
    return (
            i,
            j,
        *_shared_16x16_to_ldmatrix_32x8_layout(wmma_i, wmma_j),
        )



sch.transform_layout(block_shared_local_A, ("write", 0), index_map_B)
sch.transform_layout(block_shared_local_B, ("write", 0), index_map_B)
sch.transform_layout(block_local_C, ("read", 0), index_map_B)
write_sch(sch, log_path, "transform_layout")

init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
# sch.tensorize(sch.get_loops(init_block_b)[-2], MMA_fill_16x16_f16_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(block_shared_local_A)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_A)[-2], LDMATRIX_16x16_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(block_shared_local_B)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_B)[-2], LDMATRIX_16x16_B_INTRIN)
sch.tensorize(kernel_i, MMA_f16f16f16_INTRIN)
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

cuda_a = tvm.nd.array(np.arange(M * K).reshape((M // wmma_m, K // wmma_k, wmma_m, wmma_k)).astype("float16"), ctx)
cuda_b = tvm.nd.array(np.arange(N * K).reshape((K // wmma_k, N // wmma_n, wmma_k, wmma_n)).astype("float16"), ctx)
cuda_c = tvm.nd.array(
    np.zeros((M // wmma_m, N // wmma_m, wmma_m, wmma_n)).astype("float16"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)

num_flops = 2 * M * K * N
num_runs = 1
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
