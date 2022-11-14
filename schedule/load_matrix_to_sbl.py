import tvm
from tvm import te
import numpy as np

def intrin_load_matrix_to_slb():
    output_shape = (16, 64)
    strides_src = [64, 1]
    strides_dst = [64, 1]

    A = te.placeholder(output_shape, name="A", dtype="float32")
    C = te.compute(output_shape, lambda *i: A(*i), name="C")

    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="global",
                             strides=strides_src, data_alignment=32, offset_factor=1)
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="shared",
                             strides=strides_dst, data_alignment=32, offset_factor=1)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]

        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", 64)
        index = tx // 1

        for outer in range(0, 16):
            ib.emit(BC.vstore([outer, index], BA.vload(
                [outer, index], "float32")))

        return ib.get()
    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


M = 64
N = 64
wmma_m = 8
wmma_n = 8
wmma_k = 8

A = te.placeholder((M, N), dtype="float32", name="A")
B = te.compute((M, N), lambda *i: A(*i), name="B", )

s = te.create_schedule(B.op)
tx = te.thread_axis("threadIdx.x")
AS = s.cache_read(A, "shared", [B])
cx, ci = B.op.axis
cxo, cxi = s[B].split(cx, factor=16)
s[B].reorder(cxo, cxi, ci)
s[B].bind(ci, tx)

s[AS].compute_at(s[B], cxo)
ax, ai = AS.op.axis
print(tvm.lower(s, [A, B]))

s[AS].tensorize(ax, intrin_load_matrix_to_slb())

print(tvm.lower(s, [A, B]))

f = tvm.build(s, [A, B], "cuda")
print(f.imported_modules[0].get_source())
ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(M, N)).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros((M, N), dtype=B.dtype), ctx)
f(a, b)
# print a and b
print(a)
print(b)
