import tvm

n = 1024
k = tvm.te.reduce_axis((0, n), name='k')

A = tvm.te.placeholder((n,), name='A')
B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis=k), name='B')

s = tvm.te.create_schedule(B.op)
ko, ki = s[B].split(s[B].op.reduce_axis[0], 32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

BR = s.rfactor(B, ki)

print(tvm.lower(s, [A, B], simple_mode=True))
