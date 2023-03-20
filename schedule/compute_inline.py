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
    def main(a: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(a, (128, 128))
        B = T.alloc_buffer((128, 128))
        C = T.match_buffer(c, (128, 128))
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

block_b = sch.get_block("B")
block_c = sch.get_block("C")
sch.reverse_compute_inline(block_c)

print(sch.mod["main"].script())
