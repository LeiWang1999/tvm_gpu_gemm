# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "float16"], B: T.Buffer[(16384, 16384), "float16"], C: T.Buffer[(16384, 16384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([16384, 16384], dtype="float16", scope="shared")
    B_shared = T.alloc_buffer([16384, 16384], dtype="float16", scope="shared")
    A_shared_warp = T.alloc_buffer([16384, 16384], dtype="float16", scope="warp")
    B_shared_warp = T.alloc_buffer([16384, 16384], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([16384, 16384], dtype="float16", scope="warp")
    for i_0 in T.thread_binding(64, thread="blockIdx.y"):
        for j_0 in T.thread_binding(128, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for k_0 in T.serial(512):
                        for ax0_ax1_fused_0, ax0_ax1_fused_1 in T.grid(4, 2):
                            for ax0_ax1_fused_2 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_4 in T.vectorized(8):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(16384, i_0 * 256 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 32)
                                            v1 = T.axis.spatial(16384, k_0 * 32 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 32)
                                            T.reads(A[v0, v1])
                                            T.writes(A_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                            A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0, ax0_ax1_fused_1 in T.grid(2, 2):
                            for ax0_ax1_fused_2 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_4 in T.vectorized(8):
                                        with T.block("B_shared"):
                                            v0 = T.axis.spatial(16384, k_0 * 32 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 128)
                                            v1 = T.axis.spatial(16384, j_0 * 128 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 128)
                                            T.reads(B[v0, v1])
                                            T.writes(B_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                            B_shared[v0, v1] = B[v0, v1]
                        for i_1_1_0, j_1_1_0, k_1_0 in T.grid(8, 2, 2):
                            for ax0, ax1 in T.grid(16, 16):
                                with T.block("A_shared_warp"):
                                    v0 = T.axis.spatial(16384, i_0 * 256 + i_1_0 * 128 + i_1_1_0 * 16 + ax0)
                                    v1 = T.axis.spatial(16384, k_0 * 32 + k_1_0 * 16 + ax1)
                                    T.reads(A_shared[v0, v1])
                                    T.writes(A_shared_warp[v0, v1])
                                    A_shared_warp[v0, v1] = A_shared[v0, v1]
                            for ax0, ax1 in T.grid(16, 16):
                                with T.block("B_shared_warp"):
                                    v0 = T.axis.spatial(16384, k_0 * 32 + k_1_0 * 16 + ax0)
                                    v1 = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 32 + j_1_1_0 * 16 + ax1)
                                    T.reads(B_shared[v0, v1])
                                    T.writes(B_shared_warp[v0, v1])
                                    B_shared_warp[v0, v1] = B_shared[v0, v1]
                            for i_1_1_1, j_1_1_1, k_1_1 in T.grid(16, 16, 16):
                                with T.block("B"):
                                    vi = T.axis.spatial(16384, i_0 * 256 + i_1_0 * 128 + i_1_1_0 * 16 + i_1_1_1)
                                    vj = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 32 + j_1_1_0 * 16 + j_1_1_1)
                                    vk = T.axis.reduce(16384, k_0 * 32 + k_1_0 * 16 + k_1_1)
                                    T.reads(A_shared_warp[vi, vk], B_shared_warp[vk, vj])
                                    T.writes(C_warp[vi, vj])
                                    with T.init():
                                        C_warp[vi, vj] = T.float16(0)
                                    C_warp[vi, vj] = C_warp[vi, vj] + A_shared_warp[vi, vk] * B_shared_warp[vk, vj]
                    for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(8, 2, 16, 16):
                        with T.block("C_warp"):
                            v0 = T.axis.spatial(16384, i_0 * 256 + i_1_0 * 128 + ax0_0 * 16 + ax0_1)
                            v1 = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 32 + ax1_0 * 16 + ax1_1)
                            T.reads(C_warp[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_warp[v0, v1]
