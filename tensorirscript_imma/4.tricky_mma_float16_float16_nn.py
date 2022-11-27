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
from tvm.tir.expr import Cast, IntImm
from tvm.tir.function import TensorIntrin

log_path = "progress/tensorirscript_imma/4.tricky_mma_float16_float16_nn"
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


def get_mma_fill_intrin(dtype, local_size):
    m_dim = 16
    k_dim = 16
    n_dim = 16
    warp_size = 32
    zero = IntImm("int32", 0).astype(dtype)
    def _shared_16x16_to_ldmatrix_32x8_layout(i, j):
        return (i * 2 + j // 8, j % 8)
    # Assume M = N = 16
    index_map = _shared_16x16_to_ldmatrix_32x8_layout

    @T.prim_func
    def mma_fill_desc(a: T.handle) -> None:
        C_warp = T.match_buffer(
            a, [warp_size, local_size], dtype=dtype, scope="warp")

        with T.block("root"):
            T.reads()
            T.writes(C_warp[0:warp_size, 0:local_size])
            for i0, i1 in T.grid(m_dim, n_dim):
                with T.block("C_warp"):
                    i, j = T.axis.remap("SS", [i0, i1])
                    thread_id, local_id = T.meta_var(index_map(i, j))
                    T.reads()
                    T.writes(C_warp[thread_id, local_id])
                    C_warp[thread_id, local_id] = zero

    @T.prim_func
    def mma_fill_impl(a: T.handle) -> None:
        C_warp = T.match_buffer(
            a, [warp_size, local_size], dtype=dtype, scope="warp", offset_factor=1
        )

        with T.block("root"):
            T.reads()
            T.writes(C_warp[0:warp_size, 0:local_size])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, warp_size)

            T.evaluate(T.mma_fill(local_size, C_warp.data,
                       C_warp.elem_offset, dtype=dtype))

    return mma_fill_desc, mma_fill_impl


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


    def _shared_16x16_to_ldmatrix_32x8_layout(i, j):
        return (i * 2 + j // 8, j % 8)
        
    if k_dim == 16:
        assert dtype == "float16"

        index_map = _shared_16x16_to_ldmatrix_32x8_layout

        def shared_offset(tx, stride): 
            # stride is a constant, which means the leading dimension size.
            
            return tx * stride // 2
        
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


def get_mma_intrin(k_dim, out_dtype, b_transposed):
    m_dim = 16
    k_dim = 16
    n_dim = 16
    warp_size = 32
    local_size = (m_dim * k_dim) // warp_size
    local_size_out = (m_dim * n_dim) // 32

    def _shared_16x16_to_ldmatrix_32x8_layout(i, j):
        return (i * 2 + j // 8, j % 8)

    index_map_C = _shared_16x16_to_ldmatrix_32x8_layout

    if k_dim == 16:
        index_map_A = _shared_16x16_to_ldmatrix_32x8_layout
        index_map_B = _shared_16x16_to_ldmatrix_32x8_layout
        mma_prefix = "m16n8k16"
    else:
        assert False

    out_dtype_abbrv = {"float16": "fp16",
                       "float32": "fp32", "int32": "int32"}[out_dtype]

    if out_dtype in ["float16", "float32"]:
        in_dtype = "float16"
        in_dtype_abbrv = "fp16"
    else:
        in_dtype = "int8"
        in_dtype_abbrv = "int8"

    def maybe_cast(v):
        if out_dtype in ["float32", "int32"]:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    @T.prim_func
    def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (warp_size, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        B = T.match_buffer(
            b, (warp_size, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        C = T.match_buffer(
            c, (warp_size, local_size_out), out_dtype, align=64, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(
                C[0:warp_size, 0:local_size_out],
                A[0:warp_size, 0:local_size],
                B[0:warp_size, 0:local_size],
            )
            T.writes(C[0:warp_size, 0:local_size_out])

            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block("C"):
                    i, j, k = T.axis.remap("SSR", [i, j, k])
                    b_row_ind, b_col_ind = maybe_swap(k, j)

                    thread_id_C, local_id_C = T.meta_var(index_map_C(i, j))
                    thread_id_A, local_id_A = T.meta_var(index_map_A(i, k))
                    thread_id_B, local_id_B = T.meta_var(
                        index_map_B(b_row_ind, b_col_ind))

                    T.reads(
                        C[thread_id_C, local_id_C],
                        A[thread_id_A, local_id_A],
                        B[thread_id_B, local_id_B],
                    )
                    T.writes(C[thread_id_C, local_id_C])

                    C[thread_id_C, local_id_C] += maybe_cast(
                        A[thread_id_A, local_id_A]
                    ) * maybe_cast(B[thread_id_B, local_id_B])

    @T.prim_func
    def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (warp_size, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        B = T.match_buffer(
            b, (warp_size, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        C = T.match_buffer(
            c, (warp_size, local_size_out), out_dtype, align=64, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(
                C[0:warp_size, 0:local_size_out],
                A[0:warp_size, 0:local_size],
                B[0:warp_size, 0:local_size],
            )
            T.writes(C[0:warp_size, 0:local_size_out])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, warp_size)

            T.evaluate(
                T.ptx_mma(
                    mma_prefix,
                    "row",
                    "col",
                    in_dtype_abbrv,
                    in_dtype_abbrv,
                    out_dtype_abbrv,
                    A.data,
                    A.elem_offset + tx * convert(local_size),
                    B.data,
                    B.elem_offset + tx * convert(local_size),
                    C.data,
                    C.elem_offset + tx * convert(local_size_out),
                    False,
                    dtype=out_dtype,
                )
            )

            T.evaluate(
                T.ptx_mma(
                    mma_prefix,
                    "row",
                    "col",
                    in_dtype_abbrv,
                    in_dtype_abbrv,
                    out_dtype_abbrv,
                    A.data,
                    A.elem_offset + tx * convert(local_size),
                    B.data,
                    B.elem_offset + tx *
                    convert(local_size) + convert(local_size) // 2,
                    C.data,
                    C.elem_offset + tx *
                    convert(local_size_out) + convert(local_size_out) // 2,
                    False,
                    dtype=out_dtype,
                )
            )

    return mma_sync_desc, mma_sync_impl


_MMA_fill_16x16_f16_INTRIN = "_mma_fill_16x16_f16"
TensorIntrin.register(_MMA_fill_16x16_f16_INTRIN, *
                      get_mma_fill_intrin("float16", 8))

_MMA_f16f16f16_INTRIN = "_mma_f16f16f16"
TensorIntrin.register(_MMA_f16f16f16_INTRIN, *
                      get_mma_intrin(16, "float16", False))

_LDMATRIX_16x16_A_INTRIN = "_mma.ldmatrix_16x16_a"
TensorIntrin.register(_LDMATRIX_16x16_A_INTRIN, *
                      get_ldmatrix_intrin("float16", False, False))

_LDMATRIX_16x16_B_INTRIN = "_mma.ldmatrix_16x16_b"
TensorIntrin.register(_LDMATRIX_16x16_B_INTRIN, *
                      get_ldmatrix_intrin("float16", True, False))



VERIFY = True

M = 16384
N = 16384
K = 16384
if VERIFY:
    M = 256
    N = 256
    K = 256

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
sch.tensorize(sch.get_loops(init_block_b)[-2], _MMA_fill_16x16_f16_INTRIN)
write_sch(sch, log_path,
          "tensorize_fill")
block_shared_local_A_i, block_shared_local_A_j = sch.get_loops(block_shared_local_A)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_A)
              [-2], _LDMATRIX_16x16_A_INTRIN)
write_sch(sch, log_path,
          "tensorize_load")
block_shared_local_B_i, block_shared_local_B_j = sch.get_loops(block_shared_local_B)[-4:-2]
sch.tensorize(sch.get_loops(block_shared_local_B)
              [-2], _LDMATRIX_16x16_B_INTRIN)
sch.tensorize(kernel_i, _MMA_f16f16f16_INTRIN)

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

a_np = np.random.rand(
    M // wmma_m, K // wmma_k, wmma_m, wmma_k).astype("float16")
b_np = np.random.rand(
    K // wmma_k, N // wmma_n, wmma_k, wmma_n).astype("float16")
cuda_a = tvm.nd.array((a_np).astype("float16"), ctx)
cuda_b = tvm.nd.array((b_np).astype("float16"), ctx)
cuda_c = tvm.nd.array(
    np.zeros((M // wmma_m, N // wmma_m, wmma_m, wmma_n)).astype("float16"), ctx)

if VERIFY:
    cuda_mod(cuda_a, cuda_b, cuda_c)
    a_np = a_np.transpose((0, 2, 1, 3)).reshape(M, N)
    b_np = b_np.transpose((0, 2, 1, 3)).reshape(K, N)
    c_np = cuda_c.numpy().transpose((0, 2, 1, 3)).reshape(M, N)
    np.testing.assert_allclose(
        c_np, np.matmul(a_np.astype("float16"), b_np.astype("float16")), rtol=1e-1, atol=1e-1
    )

num_flops = 2 * M * K * N
num_runs = 1
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
