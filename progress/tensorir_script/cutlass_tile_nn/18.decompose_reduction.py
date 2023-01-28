# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "float32"], B: T.Buffer[(16384, 16384), "float32"], C: T.Buffer[(16384, 16384), "float32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    A_local_shared = T.alloc_buffer([16384, 16384], dtype="float32", scope="shared")
    A_local_shared_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    B_shared = T.alloc_buffer([16384, 16384], dtype="float32", scope="shared")
    B_shared_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    C_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    for i_0 in T.thread_binding(512, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for i_1 in T.thread_binding(2, thread="vthread.y"):
                for j_1 in T.thread_binding(2, thread="vthread.x"):
                    for i_2 in T.thread_binding(4, thread="threadIdx.y"):
                        for j_2 in T.thread_binding(32, thread="threadIdx.x"):
                            for i_3_init, j_3_init in T.grid(4, 4):
                                with T.block("B_init"):
                                    vi = T.axis.spatial(16384, i_0 * 32 + i_1 * 16 + i_2 * 4 + i_3_init)
                                    vj = T.axis.spatial(16384, j_0 * 256 + j_1 * 128 + j_2 * 4 + j_3_init)
                                    T.reads()
                                    T.writes(C_local[vi, vj])
                                    C_local[vi, vj] = T.float32(0)
                            for k_0 in T.serial(512):
                                for ax0_ax1_0_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_0_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_0_fused_0 in T.serial(2):
                                            for ax1_1 in T.vectorized(4):
                                                with T.block("A_local"):
                                                    v0 = T.axis.spatial(16384, i_0 * 32 + (ax0_ax1_0_fused_0 * 128 + ax0_ax1_0_fused_1 * 32 + ax0_ax1_0_fused_2) // 8)
                                                    v1 = T.axis.spatial(16384, k_0 * 32 + (ax0_ax1_0_fused_0 * 128 + ax0_ax1_0_fused_1 * 32 + ax0_ax1_0_fused_2) % 8 * 4 + ax1_1)
                                                    T.reads(A[v0, v1])
                                                    T.writes(A_local[v0, v1])
                                                    A_local[v0, v1] = A[v0, v1]
                                            for ax0 in T.serial(4):
                                                with T.block("A_local_shared"):
                                                    v0 = T.axis.spatial(16384, i_0 * 32 + (ax0_ax1_0_fused_0 * 128 + ax0_ax1_0_fused_1 * 32 + ax0_ax1_0_fused_2) // 8)
                                                    v1 = T.axis.spatial(16384, k_0 * 32 + (ax0_ax1_0_fused_0 * 128 + ax0_ax1_0_fused_1 * 32 + ax0_ax1_0_fused_2) % 8 * 4 + ax0)
                                                    T.reads(A_local[v0, v1])
                                                    T.writes(A_local_shared[v1, v0])
                                                    A_local_shared[v1, v0] = A_local[v0, v1]
                                for ax0_ax1_fused_0 in T.serial(16):
                                    for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                        for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_3 in T.vectorized(4):
                                                with T.block("B_shared"):
                                                    v0 = T.axis.spatial(16384, k_0 * 32 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 256)
                                                    v1 = T.axis.spatial(16384, j_0 * 256 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 256)
                                                    T.reads(B[v0, v1])
                                                    T.writes(B_shared[v0, v1])
                                                    B_shared[v0, v1] = B[v0, v1]
                                for k_1 in T.serial(32):
                                    for ax0 in T.serial(4):
                                        with T.block("A_local_shared_local"):
                                            v0 = T.axis.spatial(16384, i_0 * 32 + i_1 * 16 + i_2 * 4 + ax0)
                                            v1 = T.axis.spatial(16384, k_0 * 32 + k_1)
                                            T.reads(A_local_shared[v1, v0])
                                            T.writes(A_local_shared_local[v0, v1])
                                            A_local_shared_local[v0, v1] = A_local_shared[v1, v0]
                                    for ax0 in T.serial(4):
                                        with T.block("B_shared_local"):
                                            v0 = T.axis.spatial(16384, k_0 * 32 + k_1)
                                            v1 = T.axis.spatial(16384, j_0 * 256 + j_1 * 128 + j_2 * 4 + ax0)
                                            T.reads(B_shared[v0, v1])
                                            T.writes(B_shared_local[v0, v1])
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for i_3, j_3 in T.grid(4, 4):
                                        with T.block("B_update"):
                                            vi = T.axis.spatial(16384, i_0 * 32 + i_1 * 16 + i_2 * 4 + i_3)
                                            vj = T.axis.spatial(16384, j_0 * 256 + j_1 * 128 + j_2 * 4 + j_3)
                                            vk = T.axis.reduce(16384, k_0 * 32 + k_1)
                                            T.reads(C_local[vi, vj], A_local_shared_local[vi, vk], B_shared_local[vk, vj])
                                            T.writes(C_local[vi, vj])
                                            C_local[vi, vj] = C_local[vi, vj] + A_local_shared_local[vi, vk] * B_shared_local[vk, vj]
                            for ax0, ax1 in T.grid(4, 4):
                                with T.block("C_local"):
                                    v0 = T.axis.spatial(16384, i_0 * 32 + i_1 * 16 + i_2 * 4 + ax0)
                                    v1 = T.axis.spatial(16384, j_0 * 256 + j_1 * 128 + j_2 * 4 + ax1)
                                    T.reads(C_local[v0, v1])
                                    T.writes(C[v0, v1])
                                    C[v0, v1] = C_local[v0, v1]
