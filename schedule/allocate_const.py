import tvm
from tvm import te
import numpy as np
import tvm.testing
from tvm.script import tir as T
from tvm.tir import TensorIntrin


M=64
N=64
S=8
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
                B[vi, vj] = T.cast(S, "int8") * A[vi, vj]


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

block_b = sch.get_block("B")
Buffer_A = sch.cache_read(block_b, 0, "global")

sch.bind(sch.get_loops(block_b)[0], "blockIdx.x")
sch.bind(sch.get_loops(block_b)[1], "threadIdx.x")

sch.bind(sch.get_loops(Buffer_A)[0], "blockIdx.x")
sch.bind(sch.get_loops(Buffer_A)[1], "threadIdx.x")

print(sch.mod["main"].script())

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
print(cuda_mod.imported_modules[0].get_source())
