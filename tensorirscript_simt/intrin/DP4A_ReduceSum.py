from tvm.tir import TensorIntrin
from tvm.script import tir as T

@T.prim_func
def dp4a_desc(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="local"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4])
        T.writes(C[0])
        for i in range(0, 4):
            with T.block("update"):
                vi = T.axis.remap("R", [i])
                C[0] = C[0] + T.cast(A[vi], "int32")


@T.prim_func
def dp4a_impl(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="local"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4])
        T.writes(C[0])
        # B = T.alloc_buffer((1,), "int8x4", offset_factor=1, align=4, scope="local")
        C[0] = T.call_pure_extern(
            "__dp4a", A.vload([0], "int8x4"), 1, C[0], dtype="int32"
        )


DP4A_REDUCE_SUM_INTRIN = "DP4A_ReduceSum"

TensorIntrin.register(DP4A_REDUCE_SUM_INTRIN, dp4a_desc, dp4a_impl)
