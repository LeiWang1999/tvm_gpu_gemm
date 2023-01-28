# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    PA = T.alloc_buffer([16384], dtype="int32")
    A_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    B_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for k_0 in T.serial(256):
                        for ax0_ax1_fused_0 in T.serial(131072):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(16):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(16384, (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 16 + ax0_ax1_fused_3) // 16384)
                                            v1 = T.axis.spatial(16384, (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 16 + ax0_ax1_fused_3) % 16384)
                                            T.reads(A[v0, v1])
                                            T.writes(A_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 0]]})
                                            A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(8):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(16):
                                        with T.block("B_shared"):
                                            v0 = T.axis.spatial(16384, j_0 * 256 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 16 + ax0_ax1_fused_3) // 64)
                                            v1 = T.axis.spatial(16384, k_0 * 64 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 16 + ax0_ax1_fused_3) % 64)
                                            T.reads(B[v0, v1])
                                            T.writes(B_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 0]]})
                                            B_shared[v0, v1] = B[v0, v1]
                        for i_1_1, j_1_1, k_1 in T.grid(64, 64, 64):
                            with T.block("B"):
                                vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + i_1_1)
                                vj = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + j_1_1)
                                vk = T.axis.reduce(16384, k_0 * 64 + k_1)
                                T.reads(A_shared[vi, vk], B_shared[vj, vk])
                                T.writes(C[vi, vj])
                                with T.init():
                                    C[vi, vj] = 0
                                C[vi, vj] = C[vi, vj] + T.Cast("int32", A_shared[vi, vk]) * T.Cast("int32", B_shared[vj, vk])
                        for ax0, ax1 in T.grid(16384, 16384):
                            with T.block("Pre_compute_A"):
                                vi, vk = T.axis.remap("SR", [ax0, ax1])
                                T.reads(A_shared[vi, vk])
                                T.writes(PA[vi])
                                with T.init():
                                    PA[vi] = 0
                                PA[vi] = PA[vi] + 1 * T.Cast("int32", A_shared[vi, vk])
