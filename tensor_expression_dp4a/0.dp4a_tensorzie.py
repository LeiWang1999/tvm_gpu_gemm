import tvm
from tvm import te
import numpy as np
import tvm.testing
from intrin_dp4a import dp4a


_dtype = "int8"
device = "cuda"
dev = tvm.device(device, 0)
intrin_dp4a = dp4a("local", "local", "local", ("int8", "int8"))


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


def test_gemm():
    # graph
    nn = 16384
    n = te.var("n")
    n = tvm.runtime.convert(nn)
    m, l = n, n
    A = te.placeholder((n, l), dtype=_dtype, name="A")
    B = te.placeholder((m, l), dtype=_dtype, name="B")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute((m, n), lambda ii, jj: te.sum(
        A[jj, k].astype("int32") * B[ii, k].astype("int32"), axis=k), name="C")

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
    Vthread_Size_X = 4
    Vthread_Size_Y = 2

    BK = 32

    bx, xi = s[C].split(C.op.axis[0], nparts=Grid_Size_X)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/2.split_i.cu")
    by, yi = s[C].split(C.op.axis[1], nparts=Grid_Size_Y)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/3.split_j.cu")

    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].reorder(by, bx, yi, xi)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/4.bind_block.cu")
    tyz, yi = s[C].split(yi, nparts=Vthread_Size_Y)
    ty, yi = s[C].split(yi, nparts=Block_Size_Y)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/4.1.split_yi.cu")
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/4.2.split_xi.cu")
    txz, xi = s[C].split(xi, nparts=Vthread_Size_X)
    tx, xi = s[C].split(xi, nparts=Block_Size_X)
    # xi, tx = s[C].split(xi, factor=Block_Size_X * VECTOR_FETCH_SIZE)
    # tx, vxi = s[C].split(tx, nparts=Block_Size_X)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)

    s[C].reorder(tyz, txz, ty, tx, yi, xi)
    # s[C].reorder(ty, tx, yi, xi)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/5.bind_thread.cu")

    s[CC].compute_at(s[C], tx)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/6.CC_compute_at.cu")

    # thread block tiling
    ko, ki = s[CC].split(k, factor=BK*4)
    ki, kt = s[CC].split(ki, nparts=BK)
    yc, xc = s[CC].op.axis
    s[CC].reorder(yc, xc, ko, ki, kt)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/6.5.split_k.cu")
    s[CC].tensorize(kt, intrin_dp4a)
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

    aa_yi, aa_xi = s[AA].op.axis
    aa_yi, aa_ty = s[AA].split(aa_yi, factor=Block_Size_Y)
    s[AA].reorder(aa_ty, aa_yi, aa_xi)
    aa_yi_xi_fused = s[AA].fuse(aa_xi, aa_yi)
    aa_xi, aa_tx = s[AA].split(
        aa_yi_xi_fused, factor=Block_Size_X * 16)
    aa_tx, aa_vi = s[AA].split(aa_tx, nparts=Block_Size_X)
    # s[AA].reorder(aa_ty, aa_tx, aa_yi, aa_vi)

    s[AA].bind(aa_ty, thread_y)
    s[AA].bind(aa_tx, thread_x)
    s[AA].vectorize(aa_vi)

    bb_yi, bb_xi = s[BB].op.axis
    bb_yi, bb_ty = s[BB].split(bb_yi, factor=Block_Size_Y)
    s[BB].reorder(bb_ty, bb_yi, bb_xi)
    bb_yi_xi_fused = s[BB].fuse(bb_xi, bb_yi)
    bb_xi, bb_tx = s[BB].split(
        bb_yi_xi_fused, factor=Block_Size_X * 16)
    bb_tx, bb_vi = s[BB].split(bb_tx, nparts=Block_Size_X)
    # s[BB].reorder(bb_ty, bb_tx, bb_yi, bb_xi, bb_vi)
    s[BB].bind(bb_ty, thread_y)
    s[BB].bind(bb_tx, thread_x)
    s[BB].vectorize(bb_vi)

    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/9.shared_load_bind.cu")

    al_yi, al_xi = s[AL].op.axis
    # al_fused = s[AL].fuse(al_yi, al_xi)
    # _, al_vi = s[AL].split(al_fused, factor=16)
    s[AL].vectorize(al_xi)

    bl_yi, bl_xi = s[BL].op.axis
    # s[BL].fuse(bl_yi, bl_xi)
    s[BL].vectorize(bl_xi)

    s[C].vectorize(yi)

    # s[CC].tensorize(xc, intrin_dp4a)

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
    # tvm.testing.assert_allclose(c.numpy(), np.dot(b_np.T, a_np), rtol=1e1)

    num_flops = 2 * nn * nn * nn
    num_runs = 1
    timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))


if __name__ == "__main__":
    test_gemm()
