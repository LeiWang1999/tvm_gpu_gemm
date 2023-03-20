import tvm
import tvm.testing
from tvm.script import tir as T

M=64
N=64
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, N], dtype="float16")
        B = T.match_buffer(b, [N, N], dtype="float16")
        for i, j in T.grid(M, N):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj]


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

block_b = sch.get_block("B")
Buffer_A = sch.cache_read(block_b, 0, "shared")
sch.transform_layout(block_b, ("read", 0), lambda i, j: (i // 4, j // 4, i % 4, j % 4))
i, j = sch.get_loops(block_b)
sch.compute_at(Buffer_A, i)
ii, i = sch.split(i, factors=[16, None])
sch.bind(ii, "blockIdx.x")
sch.bind(i, "threadIdx.x")
sch.cache_index(block_b, 0)
print(sch.mod)

shared_loop = sch.get_loops(Buffer_A)[-1]
shared_loop, shared_loop_v = sch.split(shared_loop, factors=[None, 8])

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
print(cuda_mod.imported_modules[0].get_source())
