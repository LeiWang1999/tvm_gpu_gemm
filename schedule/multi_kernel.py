import tvm
from tvm import te


M = 64
N = 64
wmma_m = 8
wmma_n = 8
wmma_k = 8

A = te.placeholder((M, N), dtype="float32", name="A")
B = te.compute((M, N), lambda *i: A(*i), name="B", )

s = te.create_schedule(B.op)
bx = te.thread_axis("blockIdx.x")
tx = te.thread_axis("threadIdx.x")
inputA_bx = te.thread_axis("blockIdx.x")
inputA_tx = te.thread_axis("threadIdx.x")

A_Buffer = s.cache_read(A, "global", [B])
AS = s.cache_read(A_Buffer, "shared", [B])
cx, ci = B.op.axis
cxo, cxi = s[B].split(cx, factor=16)
s[B].reorder(cxo, cxi, ci)
s[B].bind(cxo, tx)

s[AS].compute_at(s[B], cxo)

# s_buffer = te.create_schedule(A_Buffer.op)
# i, j = s_buffer[A_Buffer].op.axis
# s_buffer[A_Buffer].bind(i, bx)
# s_buffer[A_Buffer].bind(j, tx)

print(tvm.lower(s, [A, B]))

s_A_Buffer_i, s_A_Buffer_j = s[A_Buffer].op.axis

s[A_Buffer].bind(s_A_Buffer_i, inputA_bx)
s[A_Buffer].bind(s_A_Buffer_j, inputA_tx)


# ax, ai = AS.op.axis
# # s[AS].tensorize(ax, intrin_load_matrix_to_slb())
# i, j, kernel_i, kernel_j = s[AS].transform_layout(
#     lambda i, j: [i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n])

print(tvm.lower(s, [A, B]))

f = tvm.build(s, [A, B], "cuda")
print(f.imported_modules[0].get_source())