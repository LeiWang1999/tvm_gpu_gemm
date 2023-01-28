# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    B_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    A_shared_warp = T.alloc_buffer([512, 1024, 32, 16], dtype="int8", scope="warp")
    B_shared_warp = T.alloc_buffer([512, 1024, 32, 16], dtype="int8", scope="warp")
    C_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="int32", scope="warp")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for i_1_1_0_init, j_1_1_0_init, i_1_1_1_init, j_1_1_1_init in T.grid(4, 4, 16, 16):
                        with T.block("B_init"):
                            vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + i_1_1_0_init * 16 + i_1_1_1_init)
                            vj = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + j_1_1_0_init * 16 + j_1_1_1_init)
                            T.reads()
                            T.writes(C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2])
                            C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2] = 0
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
                                                T.writes(A_shared[v0, v1])
                                                A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(4):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_ax1_fused_2 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_4 in T.vectorized(16):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(16384, j_0 * 256 + (ax0_ax1_fused_0 * 4096 + ax0_ax1_fused_1 * 2048 + ax0_ax1_fused_2 * 512 + ax0_ax1_fused_3 * 16 + ax0_ax1_fused_4) // 64)
                                                v1 = T.axis.spatial(16384, k_0 * 64 + (ax0_ax1_fused_0 * 4096 + ax0_ax1_fused_1 * 2048 + ax0_ax1_fused_2 * 512 + ax0_ax1_fused_3 * 16 + ax0_ax1_fused_4) % 64)
                                                T.reads(B[v0, v1])
                                                T.writes(B_shared[v0, v1])
                                                B_shared[v0, v1] = B[v0, v1]
                        for k_1_0 in T.serial(2):
                            for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(2, 2, 32, 16):
                                with T.block("A_shared_warp"):
                                    v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + ax0_0 * 32 + ax0_1)
                                    v1 = T.axis.spatial(16384, k_0 * 64 + k_1_0 * 32 + ax1_0 * 16 + ax1_1)
                                    T.reads(A_shared[v0, v1])
                                    T.writes(A_shared_warp[v0 // 32, v1 // 16, v1 % 8 * 4 + v0 % 16 // 4, v1 % 16 // 8 * 8 + v0 % 32 // 16 * 4 + v0 % 4])
                                    A_shared_warp[v0 // 32, v1 // 16, v1 % 8 * 4 + v0 % 16 // 4, v1 % 16 // 8 * 8 + v0 % 32 // 16 * 4 + v0 % 4] = A_shared[v0, v1]
                            for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(2, 2, 32, 16):
                                with T.block("B_shared_warp"):
                                    v0 = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + ax0_0 * 32 + ax0_1)
                                    v1 = T.axis.spatial(16384, k_0 * 64 + k_1_0 * 32 + ax1_0 * 16 + ax1_1)
                                    T.reads(B_shared[v0, v1])
                                    T.writes(B_shared_warp[v0 // 32, v1 // 16, v1 % 8 * 4 + v0 % 16 // 4, v1 % 16 // 8 * 8 + v0 % 32 // 16 * 4 + v0 % 4])
                                    B_shared_warp[v0 // 32, v1 // 16, v1 % 8 * 4 + v0 % 16 // 4, v1 % 16 // 8 * 8 + v0 % 32 // 16 * 4 + v0 % 4] = B_shared[v0, v1]
                            for i_1_1_0, j_1_1_0, i_1_1_1, j_1_1_1, k_1_1 in T.grid(4, 4, 16, 16, 32):
                                with T.block("B_update"):
                                    vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + i_1_1_0 * 16 + i_1_1_1)
                                    vj = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + j_1_1_0 * 16 + j_1_1_1)
                                    vk = T.axis.reduce(16384, k_0 * 64 + k_1_0 * 32 + k_1_1)
                                    T.reads(C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2], A_shared_warp[vi // 32, vk // 16, vk % 8 * 4 + vi % 16 // 4, vk % 16 // 8 * 8 + vi % 32 // 16 * 4 + vi % 4], B_shared_warp[vj // 32, vk // 16, vk % 8 * 4 + vj % 16 // 4, vk % 16 // 8 * 8 + vj % 32 // 16 * 4 + vj % 4])
                                    T.writes(C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2])
                                    C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2] = C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2] + T.Cast("int32", A_shared_warp[vi // 32, vk // 16, vk % 8 * 4 + vi % 16 // 4, vk % 16 // 8 * 8 + vi % 32 // 16 * 4 + vi % 4]) * T.Cast("int32", B_shared_warp[vj // 32, vk // 16, vk % 8 * 4 + vj % 16 // 4, vk % 16 // 8 * 8 + vj % 32 // 16 * 4 + vj % 4])
                    for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 4, 16, 16):
                        with T.block("C_warp"):
                            v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + ax0_0 * 16 + ax0_1)
                            v1 = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + ax1_0 * 16 + ax1_1)
                            T.reads(C_warp[v0 // 16, v1 // 16, v0 % 8 * 4 + v1 % 8 // 2, v1 % 16 // 8 * 4 + v0 % 16 // 8 * 2 + v1 % 2])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_warp[v0 // 16, v1 // 16, v0 % 8 * 4 + v1 % 8 // 2, v1 % 16 // 8 * 4 + v0 % 16 // 8 * 2 + v1 % 2]
