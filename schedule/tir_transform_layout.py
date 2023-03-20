import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T

'''
    evaluate tir transform layout
'''

M = 16
N = 32

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, N], dtype="int8")
        B = T.match_buffer(b, [N, N], dtype="int8")

        for i, j in T.grid(M, N):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj]

ir_module = MyModule
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

def transform_(i, j):
    return (i // 4, j // 4, i % 4, j % 4)


def B_global_16x32_to_shared_load_16x32_layout(i, j):
    thread_id = i * 2 + j // 16
    row = (i // 8) * 8 + thread_id % 8
    col = (j % 16) + 16 * ((thread_id // 8) % 2)

    return row, col

block_b = sch.get_block("B")
sch.transform_layout(block_b, ("read", 0),
                     B_global_16x32_to_shared_load_16x32_layout, rewrite_type=2)
i, j = sch.get_loops(block_b)
sch.bind(i, "threadIdx.x")

print(sch.mod)


# build and run
# ctx = tvm.cuda(0)
# cuda_mod = tvm.build(sch.mod, target="cuda")

# print(cuda_mod.imported_modules[0].get_source())

# a_np = np.arange(M * N).reshape(M // 4, N // 4 , 4, 4).astype("int8")

# b_np = np.arange(M * N).reshape(M, N).astype("int8")

# cuda_a = tvm.nd.array((a_np).astype("int8"), ctx)
# cuda_b = tvm.nd.array((b_np).astype("int8"), ctx)

# cuda_mod(cuda_a, cuda_b)

# # print(a_np)
# # print(cuda_b.asnumpy())
