# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    PA = T.alloc_buffer([16384], dtype="int32")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for k_0, i_1_1, j_1_1, k_1 in T.grid(256, 64, 64, 64):
                        with T.block("B"):
                            vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + i_1_1)
                            vj = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + j_1_1)
                            vk = T.axis.reduce(16384, k_0 * 64 + k_1)
                            T.reads(A[vi, vk], B[vj, vk])
                            T.writes(C[vi, vj])
                            with T.init():
                                C[vi, vj] = 0
                            C[vi, vj] = C[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast("int32", B[vj, vk])
    for i, k in T.grid(16384, 16384):
        with T.block("Pre_compute_A"):
            vi, vk = T.axis.remap("SR", [i, k])
            T.reads(A[vi, vk])
            T.writes(PA[vi])
            with T.init():
                PA[vi] = 0
            PA[vi] = PA[vi] + 1 * T.Cast("int32", A[vi, vk])
