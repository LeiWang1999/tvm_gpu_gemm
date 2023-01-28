# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([2048, 256, 8, 64], dtype="int8", scope="shared")
    B_shared = T.alloc_buffer([2048, 256, 8, 64], dtype="int8", scope="shared")
    A_shared_warp = T.alloc_buffer([2048, 256, 8, 64], dtype="int8", scope="warp")
    B_shared_warp = T.alloc_buffer([2048, 256, 8, 64], dtype="int8", scope="warp")
    C_warp = T.alloc_buffer([16384, 16384], dtype="int32", scope="warp")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for k_0 in T.serial(256):
                        for ax0_ax1_fused_0 in T.serial(2):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_ax1_fused_2 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_4 in T.vectorized(16):
                                            with T.block("A_shared"):
                                                v0 = T.axis.spatial(16384, i_0 * 128 + (ax0_ax1_fused_0 * 4096 + ax0_ax1_fused_1 * 2048 + ax0_ax1_fused_2 * 512 + ax0_ax1_fused_3 * 16 + ax0_ax1_fused_4) // 64)
                                                v1 = T.axis.spatial(16384, k_0 * 64 + (ax0_ax1_fused_0 * 4096 + ax0_ax1_fused_1 * 2048 + ax0_ax1_fused_2 * 512 + ax0_ax1_fused_3 * 16 + ax0_ax1_fused_4) % 64)
                                                T.reads(A[v0, v1])
                                                T.writes(A_shared[v0 // 8, v1 // 64, v0 % 8, v1 % 64])
                                                A_shared[v0 // 8, v1 // 64, v0 % 8, v1 % 64] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(4):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_ax1_fused_2 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_4 in T.vectorized(16):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(16384, j_0 * 256 + (ax0_ax1_fused_0 * 4096 + ax0_ax1_fused_1 * 2048 + ax0_ax1_fused_2 * 512 + ax0_ax1_fused_3 * 16 + ax0_ax1_fused_4) // 64)
                                                v1 = T.axis.spatial(16384, k_0 * 64 + (ax0_ax1_fused_0 * 4096 + ax0_ax1_fused_1 * 2048 + ax0_ax1_fused_2 * 512 + ax0_ax1_fused_3 * 16 + ax0_ax1_fused_4) % 64)
                                                T.reads(B[v0, v1])
                                                T.writes(B_shared[v0 // 8, v1 // 64, v0 % 8, v1 % 64])
                                                B_shared[v0 // 8, v1 // 64, v0 % 8, v1 % 64] = B[v0, v1]
                        for k_1_0 in T.serial(2):
                            for ax0, ax1, ax2 in T.grid(8, 8, 32):
                                with T.block("A_shared_warp"):
                                    v0 = T.axis.spatial(2048, i_0 * 16 + i_1_0 * 8 + ax0)
                                    v1, v2 = T.axis.remap("SS", [k_0, ax1])
                                    v3 = T.axis.spatial(64, k_1_0 * 32 + ax2)
                                    T.reads(A_shared[v0, v1, v2, v3])
                                    T.writes(A_shared_warp[v0, v1, v2, v3])
                                    A_shared_warp[v0, v1, v2, v3] = A_shared[v0, v1, v2, v3]
                            for ax0, ax1, ax2 in T.grid(8, 8, 32):
                                with T.block("B_shared_warp"):
                                    v0 = T.axis.spatial(2048, j_0 * 32 + j_1_0 * 8 + ax0)
                                    v1, v2 = T.axis.remap("SS", [k_0, ax1])
                                    v3 = T.axis.spatial(64, k_1_0 * 32 + ax2)
                                    T.reads(B_shared[v0, v1, v2, v3])
                                    T.writes(B_shared_warp[v0, v1, v2, v3])
                                    B_shared_warp[v0, v1, v2, v3] = B_shared[v0, v1, v2, v3]
                            for i_1_1_0, j_1_1_0, i_1_1_1, j_1_1_1, k_1_1 in T.grid(4, 4, 16, 16, 32):
                                with T.block("B"):
                                    vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + i_1_1_0 * 16 + i_1_1_1)
                                    vj = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + j_1_1_0 * 16 + j_1_1_1)
                                    vk = T.axis.reduce(16384, k_0 * 64 + k_1_0 * 32 + k_1_1)
                                    T.reads(A_shared_warp[vi // 8, vk // 64, vi % 8, vk % 64], B_shared_warp[vj // 8, vk // 64, vj % 8, vk % 64])
                                    T.writes(C_warp[vi, vj])
                                    with T.init():
                                        C_warp[vi, vj] = 0
                                    C_warp[vi, vj] = C_warp[vi, vj] + T.Cast("int32", A_shared_warp[vi // 8, vk // 64, vi % 8, vk % 64]) * T.Cast("int32", B_shared_warp[vj // 8, vk // 64, vj % 8, vk % 64])
                    for ax0, ax1 in T.grid(64, 64):
                        with T.block("C_warp"):
                            v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + ax0)
                            v1 = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + ax1)
                            T.reads(C_warp[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_warp[v0, v1]
