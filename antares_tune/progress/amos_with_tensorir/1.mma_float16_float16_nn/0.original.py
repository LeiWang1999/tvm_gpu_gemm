# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(8192, 8192), "float16"], B: T.Buffer[(8192, 8192), "float16"], C: T.Buffer[(8192, 8192), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    for i, j, k in T.grid(8192, 8192, 8192):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A[vi, vk], B[vk, vj])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float16(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
