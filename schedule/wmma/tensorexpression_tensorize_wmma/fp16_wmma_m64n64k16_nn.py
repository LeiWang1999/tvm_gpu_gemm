import os
import tvm
from tvm import te
import numpy as np
import tvm.testing


def intrin_wmma_load_matrix(shape, strides_dst, strides_from, scope):
    n, m, l = shape
    if scope == "wmma.matrix_a":
        row, col = n, l
    elif scope == "wmma.matrix_b":
        row, col = l, m
    A = te.placeholder((row, col), name="A", dtype="float16")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", strides=strides_from, data_alignment=32, offset_factor=row * col
    )
    C = te.compute((row, col), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=scope, strides=strides_dst, data_alignment=32, offset_factor=row * col
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                m,
                l,
                BC.elem_offset // (row * col),
                BA.access_ptr("r"),
                col,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(shape, strides_A, strides_B, strides_C):
    n, m, l = shape
    A = te.placeholder((n, l), name="A", dtype="float16")
    B = te.placeholder((l, m), name="B", dtype="float16")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute(
        (n, m),
        lambda ii, jj: te.sum(A[ii, k].astype(
            "float") * B[k, jj].astype("float"), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", strides=strides_A,data_alignment=32, offset_factor=n * l
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", strides=strides_B,data_alignment=32, offset_factor=l * m
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        strides=strides_C,
        data_alignment=32,
        offset_factor=n * m,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    n,
                    m,
                    l,
                    BC.elem_offset // (n * m),
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // (n * m),
                    BA.data,
                    BA.elem_offset // (n * l),
                    BB.data,
                    BB.elem_offset // (l * m),
                    BC.data,
                    BC.elem_offset // (n * m),
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix(shape, strides_dst, strides_from):
    n, m, l = shape
    A = te.placeholder((n, m), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", strides=strides_from,data_alignment=32, offset_factor=n * m
    )
    C = te.compute((n, m), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="global", strides=strides_dst, data_alignment=32, offset_factor=n * m
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                n,
                m,
                l,
                BA.elem_offset // (n * m),
                BC.access_ptr("w"),
                m,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


TASK = "gemm"
USE_MANUAL_CODE = False
_dtype = "float16"


def write_code(code, path, fname):
    # if path not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)


def test_gemm():
    log_path = "progress/tensorize_wmma/fp16_wmma_m64n64k16_nn"
    # graph
    m = 1024
    n = 1024
    k = 1024

    A = te.placeholder((m, k), dtype=_dtype, name="A")
    B = te.placeholder((k, n), dtype=_dtype, name="B")
    rk = te.reduce_axis((0, k), name="rk")
    C = te.compute((m, n), lambda ii, jj: te.sum(
        A[ii, rk].astype("float32") * B[rk, jj].astype("float32"), axis=rk), name="C")

    # schedule
    s = te.create_schedule(C.op)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "0.initial.cu")

    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")

    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "1.cache.cu")

    # grid_size 16384
    # block_size 256

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    warp_size = 32
    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 4
    warp_col_tiles = 2
    wmma_m = 32
    wmma_n = 8
    wmma_k = 16
    vec = 8
    chunk = 4
    offset = 0
    offsetCS = 0

    shape = (wmma_m, wmma_n, wmma_k)
    BM = wmma_m * warp_row_tiles * block_row_warps
    BN = wmma_n * warp_col_tiles * block_col_warps


    
    _m, _n = C.op.axis
    # compute_at
    bb, kernel_i = s[C].split(_m, factor=wmma_m)
    oo, kernel_j = s[C].split(_n, factor=wmma_n)
    s[C].reorder(bb, oo, kernel_i, kernel_j)

    bb, ii = s[C].split(bb, factor=warp_row_tiles)
    bx, bb = s[C].split(bb, factor=block_row_warps)
    oo, jj = s[C].split(oo, factor=warp_col_tiles)
    by, oo = s[C].split(oo, factor=block_col_warps)
    s[C].reorder(bx, by, bb, oo, ii, jj, kernel_i, kernel_j)
    
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].bind(bb, thread_y)
    s[C].bind(oo, thread_z)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "3.bind_c.cu")

    # Schedule for wmma computation
    s[CF].compute_at(s[C], oo)

    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    (rk,) = CF.op.reduce_axis
    rk, _rk = s[CF].split(rk, factor=wmma_k)
    ko, ki = s[CF].split(rk, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _rk)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "3.bind_cf.cu")

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "5.bind_af.cu")
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    
    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_k)
    i, i_ii = s[BF].split(i, factor=wmma_n)
    s[BF].reorder(o, i, o_ii, i_ii)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "5.bind_bf.cu")

    # Schedule for A's(B's) shared memory load
    def shared_schedule(stage, shape):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        xo, xi = s[stage].split(xo, factor=shape[0])
        yo, yi = s[stage].split(yo, factor=shape[1])
        s[stage].reorder(xo, yo, xi, yi)
        
        ty, xo = s[stage].split(xo, nparts=block_row_warps)
        tz, yo = s[stage].split(yo, nparts=block_col_warps)
        fused_xi_yi = s[stage].fuse(xi, yi)
        tx, fused_xi_yi = s[stage].split(fused_xi_yi, nparts=warp_size)
        # t, vi = s[stage].split(t, factor=vec)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        # s[stage].vectorize(vi)

    shared_schedule(AS, (wmma_m, wmma_k))
    shared_schedule(BS, (wmma_k, wmma_n))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "7.shared_schedule.cu")

    # Tensorize    
    AS_align = chunk * wmma_k + offset
    BS_align = chunk * wmma_k + offset
    C_align = warp_col_tiles * block_col_warps * wmma_m + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_k, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    C_stride = [C_align, 1]
    
    s[AF].tensorize(
        b_ii, intrin_wmma_load_matrix(shape, AF_stride, AS_stride, "wmma.matrix_a"))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "8.tensorize_load_A.cu")
    s[BF].tensorize(
        o_ii, intrin_wmma_load_matrix(shape, BF_stride, BS_stride, "wmma.matrix_b"))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "8.tensorize_load_B.cu")
    s[CF].tensorize(_ii, intrin_wmma_gemm(
        shape, AF_stride, BF_stride, CF_stride))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "8.tensorize_gemm.cu")
    s[C].tensorize(kernel_i, intrin_wmma_store_matrix(
        shape, C_stride, CF_stride))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path,  "8.final_stmt.cu")

    device = 'cuda -arch=sm_80'
    dev = tvm.device(device, 0)

    f = tvm.build(s, [A, B, C], device)
    write_code(f.imported_modules[0].get_source(), log_path, "tmp.cu")
    # launch the kernel.

    a_np = np.random.uniform(size=(m, k)).astype(A.dtype)
    b_np = np.random.uniform(size=(k, n)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), dev)
    for i in range(2):
        f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), np.dot(b_np.T, a_np), rtol=1e1)

    num_flops = 2 * m * n * k
    num_runs = 3
    timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))


if __name__ == "__main__":
    test_gemm()
