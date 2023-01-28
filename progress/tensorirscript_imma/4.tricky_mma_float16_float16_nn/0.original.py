# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(196, 36, 16, 16), "float16"], B: T.Buffer[(36, 4, 16, 16), "float16"], C: T.Buffer[(1, 196, 4, 16, 16), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    for sk, ii, jj, kk, i, j, k in T.grid(1, 196, 4, 36, 16, 16, 16):
        with T.block("B"):
            vsk, vii, vjj, vkk, vi, vj, vk = T.axis.remap("SSSRSSR", [sk, ii, jj, kk, i, j, k])
            T.reads(A[vii, vkk, vi, vk], B[vkk, vjj, vk, vj])
            T.writes(C[vsk, vii, vjj, vi, vj])
            with T.init():
                C[vsk, vii, vjj, vi, vj] = T.float32(0)
            C[vsk, vii, vjj, vi, vj] = C[vsk, vii, vjj, vi, vj] + A[vii, vkk, vi, vk] * B[vkk, vjj, vk, vj]
