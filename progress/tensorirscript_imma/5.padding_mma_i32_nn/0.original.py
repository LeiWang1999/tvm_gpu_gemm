# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(256, 256), "int8"], B: T.Buffer[(256, 256), "int8"], C: T.Buffer[(256, 256), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    for i, j, k in T.grid(256, 256, 256):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A[vi, vk], B[vk, vj])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = 0
            C[vi, vj] = C[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast("int32", B[vk, vj])
