# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(512, 512, 16, 16), "float16"], B: T.Buffer[(512, 512, 16, 16), "float16"], C: T.Buffer[(512, 512, 16, 16), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    for ii, jj, kk, i, j, k in T.grid(512, 512, 512, 16, 16, 16):
        with T.block("B"):
            vii, vjj, vkk, vi, vj, vk = T.axis.remap("SSRSSR", [ii, jj, kk, i, j, k])
            T.reads(A[vii, vkk, vi, vk], B[vkk, vjj, vk, vj])
            T.writes(C[vii, vjj, vi, vj])
            with T.init():
                C[vii, vjj, vi, vj] = T.float32(0)
            C[vii, vjj, vi, vj] = C[vii, vjj, vi, vj] + A[vii, vkk, vi, vk] * B[vkk, vjj, vk, vj]
