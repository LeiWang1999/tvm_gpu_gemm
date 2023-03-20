import tvm
from tvm import te
import numpy as np
import tvm.testing
from intrin_wmma import (
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)

TASK = "gemm"
USE_MANUAL_CODE = False
_dtype = "float16"

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


def test_gemm():
    # graph
    nn = 256
    n = te.var("n")
    n = tvm.runtime.convert(nn)
    m = k = n = nn
    A = te.placeholder((m, k), dtype=_dtype, name="A")
    B = te.placeholder((n, k), dtype=_dtype, name="B")
    rk = te.reduce_axis((0, k), name="rk")
    C = te.compute((m, n), lambda ii, jj: te.sum(
        A[ii, rk].astype("float32") * B[jj, rk].astype("float32"), axis=rk), name="C")

    # schedule
    s = te.create_schedule(C.op)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/0.initial.cu")

    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")

    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/1.cache.cu")

    # grid_size 16384
    # block_size 256

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")


    warp_size = 32
    block_row_warps = 4
    block_col_warps = 2
    warp_row_tiles = 2
    warp_col_tiles = 4
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
    vec = 8
    chunk = 1
    offset = 0
    offsetCS = 0
    BM = wmma_m * warp_row_tiles * block_row_warps
    BN = wmma_n * warp_col_tiles * block_col_warps
    AS_align = chunk * wmma_k + offset
    BS_align = chunk * wmma_k + offset
    CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    
    m, n = C.op.axis
    
    by, bm = s[C].split(m, factor=BM)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/2.split_bm.cu")
    bx, bn = s[C].split(n, factor=BN)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/3.split_bn.cu")
    
    s[C].reorder(by, bx, bm, bn)
    fused_bm_bn = s[C].fuse(bm, bn)
    
    fused_bm_bn, vi = s[C].split(fused_bm_bn, factor=vec)
    fused_bm_bn, tx = s[C].split(fused_bm_bn, factor=warp_size)
    fused_bm_bn, ty = s[C].split(fused_bm_bn, factor=block_row_warps)
    fused_bm_bn, tz = s[C].split(fused_bm_bn, factor=block_col_warps)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)
    print(ty)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/4.bind_block.cu")
    
    
    
    # Schedule for wmma computation
    s[CF].compute_at(s[C], bx)
    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    (k,) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    # Schedule for  wmma_matrix_b load 
    s[BF].compute_at(s[CF], ki)
    o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(o, i, o_ii, i_ii)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/5.bind_cs.cu")

    # Schedule for A's(B's) shared memory load
    def shared_schedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_schedule(AS, AS_align)
    shared_schedule(BS, BS_align)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/6.process_shared.cu")

    # Tensorize
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_k, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    CS_stride = [CS_align, 1]
    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder(
        (wmma_m, wmma_k), name="AL_gemm", dtype="float16")
    BL_gemm = te.placeholder(
        (wmma_n, wmma_k), name="BL_gemm", dtype="float16")
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(
                "float32") * BL_gemm[jj, k_gemm].astype("float32"),
            axis=k_gemm,
        ),
        name="CL_compute")
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/6.before_tensorize.cu")

    intrin_wmma_load_matrix_a = intrin_wmma_load_matrix_A(
        AF_stride,
        AS_stride,
        shape,
        "row_major",
        (wmma_m, wmma_k),
        (wmma_m, wmma_k),
        "float16",
    )
    intrin_wmma_load_matrix_b = intrin_wmma_load_matrix_W(
        BF_stride,
        BS_stride,
        shape,
        "col_major",
        (wmma_n, wmma_k),
        (wmma_n, wmma_k),
        "float16",
    )
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_a,
    )
    s[BF].tensorize(
        o_ii,
        intrin_wmma_load_matrix_b
    )
    s[CF].tensorize(
        _ii,
        intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute,
                         AF_stride, BF_stride, CF_stride, shape),
    )

    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(
            CS_stride, CF_stride, shape, "float32", (
                wmma_m, wmma_n), (wmma_m, wmma_n)
        ),
    )
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/6.tensorize.cu")



    device = 'cuda -arch=sm_80'

    dev = tvm.device(device, 0)
    # if not dev.exist:
    #     print("Skip because %s is not enabled" % device)
    #     return
    # print("Device %s" % device)
    f = tvm.build(s, [A, B, C], device)
    write_code(f.imported_modules[0].get_source(), "tmp.cu")
    # launch the kernel.
    n, m, l = nn, nn, nn
    a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
    b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), dev)
    for i in range(2):
        f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), np.dot(b_np.T, a_np), rtol=1e1)

    num_flops = 2 * nn * nn * nn
    num_runs = 3
    timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))


if __name__ == "__main__":
    test_gemm()