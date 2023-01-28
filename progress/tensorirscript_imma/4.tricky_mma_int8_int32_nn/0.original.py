# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16, 8, 16, 32), "int8"], B: T.Buffer[(8, 16, 32, 16), "int8"], C: T.Buffer[(16, 16, 16, 16), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    for ii, jj, kk, i, j, k in T.grid(16, 16, 8, 16, 16, 32):
        with T.block("B"):
            vii, vjj, vkk, vi, vj, vk = T.axis.remap("SSRSSR", [ii, jj, kk, i, j, k])
            T.reads(A[vii, vkk, vi, vk], B[vkk, vjj, vk, vj])
            T.writes(C[vii, vjj, vi, vj])
            with T.init():
                C[vii, vjj, vi, vj] = 0
            C[vii, vjj, vi, vj] = C[vii, vjj, vi, vj] + T.Cast("int32", A[vii, vkk, vi, vk]) * T.Cast("int32", B[vkk, vjj, vk, vj])
