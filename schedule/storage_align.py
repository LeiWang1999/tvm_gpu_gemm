import tvm

n = 1024
factor = 100
offset = 8
dtype = "float32"
A = tvm.te.placeholder((n, n), dtype=dtype, name='A')
k = tvm.te.reduce_axis((0, n), name='k')
B = tvm.te.compute((n,), lambda i: tvm.te.sum(A[i, k], axis=k), name='B')

s = tvm.te.create_schedule(B.op)
AA = s.cache_read(A, "shared", [B])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[AA].storage_align(AA.op.axis[0], factor, offset)

print(tvm.lower(s, [A, B], simple_mode=True))
