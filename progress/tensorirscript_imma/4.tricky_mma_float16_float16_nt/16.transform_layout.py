# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1024, 1024, 16, 16), "float16"], B: T.Buffer[(1024, 1024, 16, 16), "float16"], C: T.Buffer[(1024, 1024, 16, 16), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="shared")
    A_shared_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    B_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="shared")
    B_shared_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    for ii_0 in T.thread_binding(128, thread="blockIdx.y"):
        for jj_0_0 in T.thread_binding(4, thread="blockIdx.z"):
            for jj_0_1 in T.thread_binding(16, thread="blockIdx.x"):
                for ii_1 in T.thread_binding(2, thread="threadIdx.y"):
                    for jj_1 in T.thread_binding(2, thread="threadIdx.z"):
                        for ii_2_init, jj_2_init, i_init, j_init in T.grid(4, 8, 16, 16):
                            with T.block("B_init"):
                                vii = T.axis.spatial(1024, ii_0 * 8 + ii_1 * 4 + ii_2_init)
                                vjj = T.axis.spatial(1024, jj_0_0 * 256 + jj_0_1 * 16 + jj_1 * 8 + jj_2_init)
                                vi, vj = T.axis.remap("SS", [i_init, j_init])
                                T.reads()
                                T.writes(C_warp[vii, vjj, vi % 8 * 4 + vj % 8 // 2, vj // 8 * 4 + vi // 8 * 2 + vj % 2])
                                C_warp[vii, vjj, vi % 8 * 4 + vj % 8 // 2, vj // 8 * 4 + vi // 8 * 2 + vj % 2] = T.float32(0)
                        for kk_0 in T.serial(512):
                            for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.serial(4):
                                        for ax0_ax1_ax2_ax3_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_ax3_fused_4 in T.vectorized(8):
                                                with T.block("A_shared"):
                                                    v0 = T.axis.spatial(1024, ii_0 * 8 + (ax0_ax1_ax2_ax3_fused_0 * 2048 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) // 512)
                                                    v1 = T.axis.spatial(1024, kk_0 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 2048 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 512 // 256)
                                                    v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 2048 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 256 // 16)
                                                    v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 2048 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 16)
                                                    T.reads(A[v0, v1, v2 % 8 * 2 + v3 // 8, v2 // 8 * 8 + v3 % 8])
                                                    T.writes(A_shared[v0, v1, v2, v3])
                                                    A_shared[v0, v1, v2, v3] = A[v0, v1, v2 % 8 * 2 + v3 // 8, v2 // 8 * 8 + v3 % 8]
                            for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.serial(8):
                                        for ax0_ax1_ax2_ax3_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_ax3_fused_4 in T.vectorized(8):
                                                with T.block("B_shared"):
                                                    v0 = T.axis.spatial(1024, jj_0_0 * 256 + jj_0_1 * 16 + (ax0_ax1_ax2_ax3_fused_0 * 4096 + ax0_ax1_ax2_ax3_fused_1 * 2048 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) // 512)
                                                    v1 = T.axis.spatial(1024, kk_0 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 4096 + ax0_ax1_ax2_ax3_fused_1 * 2048 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 512 // 256)
                                                    v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 4096 + ax0_ax1_ax2_ax3_fused_1 * 2048 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 256 // 16)
                                                    v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 4096 + ax0_ax1_ax2_ax3_fused_1 * 2048 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 16)
                                                    T.reads(B[v0, v1, v2 // 8 * 8 + v2 % 4 * 2 + v3 // 8, v2 % 8 // 4 * 8 + v3 % 8])
                                                    T.writes(B_shared[v0, v1, v2, v3])
                                                    B_shared[v0, v1, v2, v3] = B[v0, v1, v2 // 8 * 8 + v2 % 4 * 2 + v3 // 8, v2 % 8 // 4 * 8 + v3 % 8]
                            for kk_1 in T.serial(2):
                                for ax0, ax1, ax2 in T.grid(4, 16, 16):
                                    with T.block("A_shared_warp"):
                                        v0 = T.axis.spatial(1024, ii_0 * 8 + ii_1 * 4 + ax0)
                                        v1 = T.axis.spatial(1024, kk_0 * 2 + kk_1)
                                        v2, v3 = T.axis.remap("SS", [ax1, ax2])
                                        T.reads(A_shared[v0, v1, v2, v3])
                                        T.writes(A_shared_warp[v0, v1, v2 * 2 + v3 // 8, v3 % 8])
                                        A_shared_warp[v0, v1, v2 * 2 + v3 // 8, v3 % 8] = A_shared[v0, v1, v2, v3]
                                for ax0, ax1, ax2 in T.grid(8, 16, 16):
                                    with T.block("B_shared_warp"):
                                        v0 = T.axis.spatial(1024, jj_0_0 * 256 + jj_0_1 * 16 + jj_1 * 8 + ax0)
                                        v1 = T.axis.spatial(1024, kk_0 * 2 + kk_1)
                                        v2, v3 = T.axis.remap("SS", [ax1, ax2])
                                        T.reads(B_shared[v0, v1, v2, v3])
                                        T.writes(B_shared_warp[v0, v1, v2 * 2 + v3 // 8, v3 % 8])
                                        B_shared_warp[v0, v1, v2 * 2 + v3 // 8, v3 % 8] = B_shared[v0, v1, v2, v3]
                                for ii_2, jj_2, i, j, k in T.grid(4, 8, 16, 16, 16):
                                    with T.block("B_update"):
                                        vii = T.axis.spatial(1024, ii_0 * 8 + ii_1 * 4 + ii_2)
                                        vjj = T.axis.spatial(1024, jj_0_0 * 256 + jj_0_1 * 16 + jj_1 * 8 + jj_2)
                                        vkk = T.axis.reduce(1024, kk_0 * 2 + kk_1)
                                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                                        T.reads(C_warp[vii, vjj, vi % 8 * 4 + vj % 8 // 2, vj // 8 * 4 + vi // 8 * 2 + vj % 2], A_shared_warp[vii, vkk, vi * 2 + vk // 8, vk % 8], B_shared_warp[vjj, vkk, vj * 2 + vk // 8, vk % 8])
                                        T.writes(C_warp[vii, vjj, vi % 8 * 4 + vj % 8 // 2, vj // 8 * 4 + vi // 8 * 2 + vj % 2])
                                        C_warp[vii, vjj, vi % 8 * 4 + vj % 8 // 2, vj // 8 * 4 + vi // 8 * 2 + vj % 2] = C_warp[vii, vjj, vi % 8 * 4 + vj % 8 // 2, vj // 8 * 4 + vi // 8 * 2 + vj % 2] + A_shared_warp[vii, vkk, vi * 2 + vk // 8, vk % 8] * B_shared_warp[vjj, vkk, vj * 2 + vk // 8, vk % 8]
                        for ax0, ax1, ax2, ax3 in T.grid(4, 8, 16, 16):
                            with T.block("C_warp"):
                                v0 = T.axis.spatial(1024, ii_0 * 8 + ii_1 * 4 + ax0)
                                v1 = T.axis.spatial(1024, jj_0_0 * 256 + jj_0_1 * 16 + jj_1 * 8 + ax1)
                                v2, v3 = T.axis.remap("SS", [ax2, ax3])
                                T.reads(C_warp[v0, v1, v2 % 8 * 4 + v3 % 8 // 2, v3 // 8 * 4 + v2 // 8 * 2 + v3 % 2])
                                T.writes(C[v0, v1, v2, v3])
                                C[v0, v1, v2, v3] = C_warp[v0, v1, v2 % 8 * 4 + v3 % 8 // 2, v3 // 8 * 4 + v2 // 8 * 2 + v3 % 2]
