import tvm
from tvm import te
import numpy as np
import tvm.testing

TASK = "gemm"
USE_MANUAL_CODE = False
_dtype = "float32"


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


def test_gemm():
    # graph
    nn = 1024
    n = te.var("n")
    n = tvm.runtime.convert(nn)
    m, l = n, n
    A = te.placeholder((l, n), dtype=_dtype, name="A")
    B = te.placeholder((l, m), dtype=_dtype, name="B")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute((m, n), lambda ii, jj: te.sum(
        A[k, jj] * B[k, ii], axis=k), name="C")
    CP = te.compute((m, n), lambda ii, jj: C[ii, jj] + 1, name="CP")

    # schedule
    s = te.create_schedule(CP.op)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    i, j = s[C].op.axis

    s[C].bind(i, block_x)
    s[C].bind(j, thread_x)
    write_code(
        str(tvm.lower(s, [A, B, C, CP], simple_mode=True)), "1.cache.cu")
    device = "cuda"

    dev = tvm.device(device, 0)
    if not dev.exist:
        print("Skip because %s is not enabled" % device)
        return
    print("Device %s" % device)
    f = tvm.build(s, [A, B, C, CP], device)

    write_code(f.imported_modules[0].get_source(), "tmp.cu")
    # launch the kernel.
    n, m, l = nn, nn, nn
    a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
    b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), dev)
    cp = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), dev)
    for i in range(2):
        f(a, b, c, cp)
    tvm.testing.assert_allclose(c.numpy(), np.dot(b_np.T, a_np), rtol=1e1)

    num_flops = 2 * nn * nn * nn
    num_runs = 30
    timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
    t = timer_f(a, b, c, cp).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))


if __name__ == "__main__":
    test_gemm()
