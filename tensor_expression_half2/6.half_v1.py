import tvm
from tvm import te
import numpy as np
import tvm.testing

TASK = "gemm"
USE_MANUAL_CODE = False
_dtype = "float16"


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


def test_gemm():
    # graph
    nn = 16384
    n = te.var("n")
    n = tvm.runtime.convert(nn)
    m, l = n, n
    A = te.placeholder((l, n), dtype=_dtype, name="A")
    B = te.placeholder((l, m), dtype=_dtype, name="B")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute((m, n), lambda ii, jj: te.sum(
        A[k, jj] * B[k, ii], axis=k), name="C")

    # schedule
    s = te.create_schedule(C.op)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/0.initial.cu")

    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/1.cache.cu")

    # grid_size 16384
    # block_size 256

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_xz = te.thread_axis("vthread", name="vx")
    thread_yz = te.thread_axis("vthread", name="vy")

    Grid_Size_X = 128
    Grid_Size_Y = 128
    Block_Size_X = 16
    Block_Size_Y = 16

    BK = 16

    bx, xi = s[C].split(C.op.axis[0], nparts=Grid_Size_X)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/2.split_i.cu")
    by, yi = s[C].split(C.op.axis[1], nparts=Grid_Size_Y)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/3.split_j.cu")

    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].reorder(by, bx, xi, yi)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/4.bind_block.cu")
    tyz, yi = s[C].split(yi, nparts=2)
    ty, yi = s[C].split(yi, nparts=Block_Size_Y)
    txz, xi = s[C].split(xi, nparts=2)
    tx, xi = s[C].split(xi, nparts=Block_Size_X)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(tyz, txz, ty, tx, xi, yi)

    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/5.bind_thread.cu")

    s[CC].compute_at(s[C], tx)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/6.CC_compute_at.cu")

    # thread block tiling
    ko, ki = s[CC].split(k, factor=BK)
    yc, xc = s[CC].op.axis
    s[CC].reorder(ko, ki, yc, xc)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/7.split_reorder_k.cu")
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[AL].compute_at(s[CC], ki)
    s[BL].compute_at(s[CC], ki)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/8.compute_at_done.cu")

    aa_yi, aa_ty = s[AA].split(s[AA].op.axis[0], factor=Block_Size_Y)
    aa_xi, aa_tx = s[AA].split(s[AA].op.axis[1], factor=Block_Size_X * 4)
    aa_tx, aa_vi = s[AA].split(aa_tx, nparts=Block_Size_X)
    s[AA].reorder(aa_ty, aa_tx, aa_yi, aa_xi, aa_vi)
    s[AA].bind(aa_ty, thread_y)
    s[AA].bind(aa_tx, thread_x)
    # s[AA].vectorize(aa_vi)

    bb_yi, bb_ty = s[BB].split(s[BB].op.axis[0], factor=Block_Size_Y)
    bb_xi, bb_tx = s[BB].split(s[BB].op.axis[1], factor=Block_Size_X * 4)
    bb_tx, bb_vi = s[BB].split(bb_tx, nparts=Block_Size_X)
    s[BB].reorder(bb_ty, bb_tx, bb_yi, bb_xi, bb_vi)
    s[BB].bind(bb_ty, thread_y)
    s[BB].bind(bb_tx, thread_x)
    # s[BB].vectorize(bb_vi)

    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/9.shared_load_bind.cu")

    # al_yi, al_xi = s[AL].op.axis
    # s[AL].vectorize(al_xi)
    # bl_yi, bl_xi = s[BL].op.axis
    # s[BL].vectorize(bl_xi)

    device = "cuda"

    dev = tvm.device(device, 0)
    if not dev.exist:
        print("Skip because %s is not enabled" % device)
        return
    print("Device %s" % device)
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
    num_runs = 1
    timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))


if __name__ == "__main__":
    test_gemm()
