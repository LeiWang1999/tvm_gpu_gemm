# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    PA = T.alloc_buffer([16384], dtype="int32", scope="shared")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for k_0 in T.serial(256):
                for i_1_0, j_1_0, i_1_1, j_1_1, k_1 in T.grid(2, 4, 64, 64, 64):
                    with T.block("B"):
                        vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + i_1_1)
                        vj = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + j_1_1)
                        vk = T.axis.reduce(16384, k_0 * 64 + k_1)
                        T.reads(A[vi, vk], B[vj, vk])
                        T.writes(C[vi, vj])
                        with T.init():
                            C[vi, vj] = 0
                        C[vi, vj] = C[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast("int32", B[vj, vk])
                for ax0, ax1 in T.grid(128, 64):
                    with T.block("Pre_compute_A"):
                        vi = T.axis.spatial(16384, i_0 * 128 + ax0)
                        vk = T.axis.reduce(16384, k_0 * 64 + ax1)
                        T.reads(A[vi, vk])
                        T.writes(PA[vi])
                        with T.init():
                            PA[vi] = 0
                        PA[vi] = PA[vi] + 1 * T.Cast("int32", A[vi, vk])
