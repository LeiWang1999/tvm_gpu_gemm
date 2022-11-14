import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T


@T.prim_func
def ldslb_desc(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (16, 64), "float32",
                       offset_factor=64, scope="global")
    B = T.match_buffer(b, (16, 64), "float32",
                       offset_factor=64, scope="shared")

    with T.block("root"):
        T.reads(A[0:16, 0:64])
        T.writes(B[0:16, 0:64])
        for i, j in T.grid(16, 64):
            with T.block("update"):
                vii, vjj  = T.axis.remap("SS", [i, j])
                B[vii, vjj] = A[vii, vjj]


@T.prim_func
def ldslb_impl(a: T.handle, b: T.handle) -> None:
    sa = T.var("int32")
    sb = T.var("int32")
    A = T.match_buffer(a, (16, 64), "float32", offset_factor=16,
                       strides=[sa, 1], scope="global")
    B = T.match_buffer(b, (16, 64), "float32", offset_factor=16,
                       strides=[sb, 1], scope="shared")

    with T.block("root"):
        T.reads(A[0:16, 0:64])
        T.writes(B[0:16, 0:64])
        for j in T.thread_binding(0, 64, "threadIdx.x"):
            for i in T.serial(0, 16):
                with T.block("update"):
                    vii, vjj  = T.axis.remap("SS", [i, j])
                    B[vii, vjj] = A[vii, vjj]


tvm.tir.TensorIntrin.register("ldslb", ldslb_desc, ldslb_impl)


M = 64
N = 64

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, N])
        B = T.match_buffer(b, [M, N])
        for i, j in T.grid(M, N):
            with T.block("B"):
                vi, vj  = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj]


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module)

block_b = sch.get_block("B")
A_shared = sch.cache_read(block_b, "A","shared")

co, ci = sch.get_loops(block_b)
sch.bind(co, "blockIdx.x")
sch.bind(ci, "threadIdx.x")

ao, ai = sch.get_loops(A_shared)
ao_o, ao_i = sch.split(ao, factors=[None, 16])
sch.tensorize(ao_i, "ldslb")
# sch.compute_at(A_shared, cxo)
# ax, ai = 
print(sch.mod["main"].script())

# print(sch.get_loops(A_shared)[-2])
# sch.tensorize(sch.get_loops(A_shared)[-2], "ldslb")
# print(sch.mod["main"].script())

cuda_mod = tvm.build(sch.mod, target="cuda")

ctx = tvm.cuda(0)
a = tvm.nd.array(np.random.uniform(size=(M, N)).astype("float32"), device=ctx)
b = tvm.nd.array(np.zeros((M, N)).astype("float32"), device=ctx)
cuda_mod(a, b)
# print a and b
print(a)
print(b)
