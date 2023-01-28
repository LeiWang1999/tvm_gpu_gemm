# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "float16"], B: T.Buffer[(16384, 16384), "float16"], C: T.Buffer[(16384, 16384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    for i_0, j_0, k_0, i_1, j_1, k_1 in T.grid(128, 64, 512, 128, 256, 32):
        with T.block("B"):
            vi = T.axis.spatial(16384, i_0 * 128 + i_1)
            vj = T.axis.spatial(16384, j_0 * 256 + j_1)
            vk = T.axis.reduce(16384, k_0 * 32 + k_1)
            T.reads(A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float16(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
