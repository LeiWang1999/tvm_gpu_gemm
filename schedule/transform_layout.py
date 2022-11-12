import tvm
from tvm import te
m = n = 16
A = te.placeholder([n,m])
B = te.compute(A.shape, lambda i,j: A[i,j])
s = te.create_schedule(B.op)

s[B].transform_layout(lambda i,j: [j,i])
print(tvm.lower(s, [A,B]))