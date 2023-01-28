# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(256, 256), "float16"], B: T.Buffer[(256, 256), "float16"], C: T.Buffer[(256, 256), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([256, 256], dtype="float16", scope="shared")
    B_shared = T.alloc_buffer([256, 256], dtype="float16", scope="shared")
    for i_0 in T.thread_binding(1, thread="blockIdx.y"):
        for j_0 in T.thread_binding(2, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for k_0 in T.serial(8):
                        for ax0_ax1_fused_0 in T.serial(8):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(8):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(256, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 8 + ax0_ax1_fused_3) // 32)
                                            v1 = T.axis.spatial(256, k_0 * 32 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 8 + ax0_ax1_fused_3) % 32)
                                            T.reads(A[v0, v1])
                                            T.writes(A_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                            A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(4):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(8):
                                        with T.block("B_shared"):
                                            v0 = T.axis.spatial(256, j_0 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 8 + ax0_ax1_fused_3) // 32)
                                            v1 = T.axis.spatial(256, k_0 * 32 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 8 + ax0_ax1_fused_3) % 32)
                                            T.reads(B[v0, v1])
                                            T.writes(B_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                            B_shared[v0, v1] = B[v0, v1]
                        for i_1_1, j_1_1, k_1 in T.grid(128, 32, 32):
                            with T.block("B"):
                                vi = T.axis.spatial(256, i_0 * 256 + i_1_0 * 128 + i_1_1)
                                vj = T.axis.spatial(256, j_0 * 128 + j_1_0 * 32 + j_1_1)
                                vk = T.axis.reduce(256, k_0 * 32 + k_1)
                                T.reads(A_shared[vi, vk], B_shared[vj, vk])
                                T.writes(C[vi, vj])
                                with T.init():
                                    C[vi, vj] = T.float16(0)
                                C[vi, vj] = C[vi, vj] + A_shared[vi, vk] * B_shared[vj, vk]
