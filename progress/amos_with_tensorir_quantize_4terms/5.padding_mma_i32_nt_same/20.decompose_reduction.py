# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    PA = T.alloc_buffer([16384], dtype="int32", scope="shared")
    A_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    B_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    A_shared_warp = T.alloc_buffer([16384, 16384], dtype="int8", scope="warp")
    B_shared_warp = T.alloc_buffer([16384, 16384], dtype="int8", scope="warp")
    C_warp = T.alloc_buffer([16384, 16384], dtype="int32", scope="warp")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for ax0_0_init in T.thread_binding(4, thread="threadIdx.y"):
                for ax0_1_init in T.thread_binding(32, thread="threadIdx.x"):
                    with T.block("Pre_compute_A_init"):
                        vi = T.axis.spatial(16384, i_0 * 128 + ax0_0_init * 32 + ax0_1_init)
                        T.reads()
                        T.writes(PA[vi])
                        PA[vi] = 0
            for i_1_0_init in T.thread_binding(2, thread="threadIdx.z"):
                for j_1_0_init in T.thread_binding(4, thread="threadIdx.y"):
                    for i_1_1_0_init, j_1_1_0_init, i_1_1_1_init, j_1_1_1_init in T.grid(4, 4, 16, 16):
                        with T.block("B_init"):
                            vi = T.axis.spatial(16384, i_0 * 128 + i_1_0_init * 64 + i_1_1_0_init * 16 + i_1_1_1_init)
                            vj = T.axis.spatial(16384, j_0 * 256 + j_1_0_init * 64 + j_1_1_0_init * 16 + j_1_1_1_init)
                            T.reads()
                            T.writes(C_warp[vi, vj])
                            C_warp[vi, vj] = 0
            for k_0 in T.serial(256):
                for ax0_ax1_fused_0 in T.serial(4):
                    for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                        for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                            for ax0_ax1_fused_3 in T.vectorized(16):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(16384, i_0 * 128 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 16 + ax0_ax1_fused_3) // 64)
                                    v1 = T.axis.spatial(16384, k_0 * 64 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 16 + ax0_ax1_fused_3) % 64)
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
                for i_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                    for j_1_0 in T.thread_binding(4, thread="threadIdx.y"):
                        for i_1_1_0, j_1_1_0, k_1_0 in T.grid(4, 4, 2):
                            for ax0, ax1 in T.grid(16, 32):
                                with T.block("A_shared_warp"):
                                    v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + i_1_1_0 * 16 + ax0)
                                    v1 = T.axis.spatial(16384, k_0 * 64 + k_1_0 * 32 + ax1)
                                    T.reads(A_shared[v0, v1])
                                    T.writes(A_shared_warp[v0, v1])
                                    A_shared_warp[v0, v1] = A_shared[v0, v1]
                            for ax0, ax1 in T.grid(16, 32):
                                with T.block("B_shared_warp"):
                                    v0 = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + j_1_1_0 * 16 + ax0)
                                    v1 = T.axis.spatial(16384, k_0 * 64 + k_1_0 * 32 + ax1)
                                    T.reads(B_shared[v0, v1])
                                    T.writes(B_shared_warp[v0, v1])
                                    B_shared_warp[v0, v1] = B_shared[v0, v1]
                            for i_1_1_1, j_1_1_1, k_1_1 in T.grid(16, 16, 32):
                                with T.block("B_update"):
                                    vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + i_1_1_0 * 16 + i_1_1_1)
                                    vj = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + j_1_1_0 * 16 + j_1_1_1)
                                    vk = T.axis.reduce(16384, k_0 * 64 + k_1_0 * 32 + k_1_1)
                                    T.reads(C_warp[vi, vj], A_shared_warp[vi, vk], B_shared_warp[vj, vk])
                                    T.writes(C_warp[vi, vj])
                                    C_warp[vi, vj] = C_warp[vi, vj] + T.Cast("int32", A_shared_warp[vi, vk]) * T.Cast("int32", B_shared_warp[vj, vk])
                        for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 4, 16, 16):
                            with T.block("C_warp"):
                                v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + ax0_0 * 16 + ax0_1)
                                v1 = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + ax1_0 * 16 + ax1_1)
                                T.reads(C_warp[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_warp[v0, v1]
                for ax0_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for ax0_1 in T.thread_binding(32, thread="threadIdx.x"):
                        for ax1_0 in T.thread_binding(2, thread="threadIdx.z"):
                            for ax1_1 in T.serial(32):
                                with T.block("Pre_compute_A_update"):
                                    vi = T.axis.spatial(16384, i_0 * 128 + ax0_0 * 32 + ax0_1)
                                    vk = T.axis.reduce(16384, k_0 * 64 + ax1_0 * 32 + ax1_1)
                                    T.reads(PA[vi], A_shared[vi, vk])
                                    T.writes(PA[vi])
                                    PA[vi] = PA[vi] + 1 * T.Cast("int32", A_shared[vi, vk])
