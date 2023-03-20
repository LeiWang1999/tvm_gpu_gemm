import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
from tvm.tir import TensorIntrin


M=64
N=64
S=8
# dtype = "float32"
dtype = "float16"
# dtype = "float32"
# dtype = "float32"
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, N], dtype=dtype)
        B = T.match_buffer(b, [N], dtype=dtype)
        for i, j in T.grid(M, N):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                # T.reads(B[vj], A[vi, vj])
                # T.writes([B[vj]])
                # with T.init():
                #     B[vj] = 0
                T.call_intrin(dtype, "tir.atomic_add", T.address_of(B[vj]), A[vi, vj])


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

block_b = sch.get_block("B")

sch.bind(sch.get_loops(block_b)[1], "blockIdx.x")
# sch.bind(sch.get_loops(block_b)[0], "threadIdx.x")

print(sch.mod["main"].script())

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
print(cuda_mod.imported_modules[0].get_source())
