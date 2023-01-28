# from tvm.script import tir as T
@T.prim_func
def func(AT: T.Buffer[(16384, 16384), "float32"], B: T.Buffer[(16384, 16384), "float32"], C: T.Buffer[(16384, 16384), "float32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    AT_shared = T.alloc_buffer([16384, 16384], dtype="float32", scope="shared")
    AT_shared_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    B_shared = T.alloc_buffer([16384, 16384], dtype="float32", scope="shared")
    B_shared_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    C_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(128, thread="blockIdx.x"):
            for i_1 in T.thread_binding(2, thread="vthread.y"):
                for j_1 in T.thread_binding(2, thread="vthread.x"):
                    for i_2 in T.thread_binding(16, thread="threadIdx.y"):
                        for j_2 in T.thread_binding(16, thread="threadIdx.x"):
                            for i_3_init, j_3_init in T.grid(4, 4):
                                with T.block("B_init"):
                                    vi = T.axis.spatial(16384, i_0 * 128 + i_1 * 64 + i_2 * 4 + i_3_init)
                                    vj = T.axis.spatial(16384, j_0 * 128 + j_1 * 64 + j_2 * 4 + j_3_init)
                                    T.reads()
                                    T.writes(C_local[vi, vj])
                                    C_local[vi, vj] = T.float32(0)
                            for k_0 in T.serial(1024):
                                for ax0_ax1_fused_0 in T.serial(2):
                                    for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_fused_2 in T.thread_binding(16, thread="threadIdx.x"):
                                            for ax0_ax1_fused_3 in T.vectorized(4):
                                                with T.block("AT_shared"):
                                                    v0 = T.axis.spatial(16384, k_0 * 16 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 64 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 128)
                                                    v1 = T.axis.spatial(16384, i_0 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 64 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 128)
                                                    T.reads(AT[v0, v1])
                                                    T.writes(AT_shared[v0, v1])
                                                    AT_shared[v0, v1] = AT[v0, v1]
                                for ax0_ax1_fused_0 in T.serial(2):
                                    for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_fused_2 in T.thread_binding(16, thread="threadIdx.x"):
                                            for ax0_ax1_fused_3 in T.vectorized(4):
                                                with T.block("B_shared"):
                                                    v0 = T.axis.spatial(16384, k_0 * 16 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 64 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 128)
                                                    v1 = T.axis.spatial(16384, j_0 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 64 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 128)
                                                    T.reads(B[v0, v1])
                                                    T.writes(B_shared[v0, v1])
                                                    B_shared[v0, v1] = B[v0, v1]
                                for k_1 in T.serial(16):
                                    for ax0 in T.serial(4):
                                        with T.block("AT_shared_local"):
                                            v0 = T.axis.spatial(16384, k_0 * 16 + k_1)
                                            v1 = T.axis.spatial(16384, i_0 * 128 + i_1 * 64 + i_2 * 4 + ax0)
                                            T.reads(AT_shared[v0, v1])
                                            T.writes(AT_shared_local[v0, v1])
                                            AT_shared_local[v0, v1] = AT_shared[v0, v1]
                                    for ax0 in T.serial(4):
                                        with T.block("B_shared_local"):
                                            v0 = T.axis.spatial(16384, k_0 * 16 + k_1)
                                            v1 = T.axis.spatial(16384, j_0 * 128 + j_1 * 64 + j_2 * 4 + ax0)
                                            T.reads(B_shared[v0, v1])
                                            T.writes(B_shared_local[v0, v1])
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for i_3, j_3 in T.grid(4, 4):
                                        with T.block("B_update"):
                                            vi = T.axis.spatial(16384, i_0 * 128 + i_1 * 64 + i_2 * 4 + i_3)
                                            vj = T.axis.spatial(16384, j_0 * 128 + j_1 * 64 + j_2 * 4 + j_3)
                                            vk = T.axis.reduce(16384, k_0 * 16 + k_1)
                                            T.reads(C_local[vi, vj], AT_shared_local[vk, vi], B_shared_local[vk, vj])
                                            T.writes(C_local[vi, vj])
                                            C_local[vi, vj] = C_local[vi, vj] + AT_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for ax0, ax1 in T.grid(4, 4):
                                with T.block("C_local"):
                                    v0 = T.axis.spatial(16384, i_0 * 128 + i_1 * 64 + i_2 * 4 + ax0)
                                    v1 = T.axis.spatial(16384, j_0 * 128 + j_1 * 64 + j_2 * 4 + ax1)
                                    T.reads(C_local[v0, v1])
                                    T.writes(C[v0, v1])
                                    C[v0, v1] = C_local[v0, v1]
