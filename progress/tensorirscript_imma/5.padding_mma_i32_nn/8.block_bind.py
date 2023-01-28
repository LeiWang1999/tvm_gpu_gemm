# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(256, 256), "int8"], B: T.Buffer[(256, 256), "int8"], C: T.Buffer[(256, 256), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    for i_0 in T.thread_binding(2, thread="blockIdx.y"):
        for j_0 in T.thread_binding(1, thread="blockIdx.x"):
            for k_0, i_1, j_1, k_1 in T.grid(4, 128, 256, 64):
                with T.block("B"):
                    vi = T.axis.spatial(256, i_0 * 128 + i_1)
                    vj = T.axis.spatial(256, j_0 * 256 + j_1)
                    vk = T.axis.reduce(256, k_0 * 64 + k_1)
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = 0
                    C[vi, vj] = C[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast("int32", B[vk, vj])
