import tvm
from tvm import te
from tvm.tir.buffer import Buffer
# tvm tir decl_buffer example
A = te.placeholder((32, 32), name="A")
B = te.placeholder((32, 32), name="B")
C = te.compute((32, 32), lambda i, j: A[i, j] + B[i, j], name="C")

BA = tvm.tir.decl_buffer(
    A.shape,
    A.dtype,
    name="BA",
    scope="global",
    data_alignment=32,
    offset_factor=8,
    elem_offset=40,
    # strides=[32, 1],
)

print(BA.elem_offset)
print(BA.data_alignment)
print(BA.offset_of((0, 1)))

s = te.create_schedule(C.op)
# print lower
# print(tvm.lower(s, [A, B, C], binds={A:BA}, simple_mode=True))