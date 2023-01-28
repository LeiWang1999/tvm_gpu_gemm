# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    A_shared_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    B_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    B_local_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    B_local_local_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    B_local_local_shared_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    C_local = T.alloc_buffer([16384, 16384], dtype="int32", scope="local")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(128, thread="blockIdx.x"):
            for i_1_0_init in T.thread_binding(16, thread="threadIdx.y"):
                for j_1_0_init in T.thread_binding(16, thread="threadIdx.x"):
                    for i_1_1_init, j_1_1_init in T.grid(8, 8):
                        with T.block("B_init"):
                            vi = T.axis.spatial(16384, i_0 * 128 + i_1_0_init * 8 + i_1_1_init)
                            vj = T.axis.spatial(16384, j_0 * 128 + j_1_0_init * 8 + j_1_1_init)
                            T.reads()
                            T.writes(C_local[vi, vj])
                            C_local[vi, vj] = T.float32(0)
            for k_0 in T.serial(512):
                for ax0_ax1_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax0_ax1_fused_2 in T.vectorized(16):
                            with T.block("A_shared"):
                                v0 = T.axis.spatial(16384, i_0 * 128 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) // 32)
                                v1 = T.axis.spatial(16384, k_0 * 32 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) % 32)
                                T.reads(A[v0, v1])
                                T.writes(A_shared[v0, v1])
                                A_shared[v0, v1] = A[v0, v1]
                for ax0_0_ax1_0_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for ax0_0_ax1_0_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax0_1 in T.serial(4):
                            for ax1_1 in T.vectorized(4):
                                with T.block("B_local"):
                                    v0 = T.axis.spatial(16384, k_0 * 32 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) // 32 * 4 + ax0_1)
                                    v1 = T.axis.spatial(16384, j_0 * 128 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) % 32 * 4 + ax1_1)
                                    T.reads(B[v0, v1])
                                    T.writes(B_local[v0, v1])
                                    B_local[v0, v1] = B[v0, v1]
                for ax0_0_ax1_0_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for ax0_0_ax1_0_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax0_1, ax1_1 in T.grid(4, 4):
                            with T.block("B_local_local"):
                                v0 = T.axis.spatial(16384, k_0 * 32 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) // 32 * 4 + ax0_1)
                                v1 = T.axis.spatial(16384, j_0 * 128 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) % 32 * 4 + ax1_1)
                                T.reads(B_local[v0, v1])
                                T.writes(B_local_local[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4])
                                B_local_local[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4] = B_local[v0, v1]
                for ax0_0_ax1_0_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for ax0_0_ax1_0_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax1_1 in T.serial(4):
                            for ax0_1 in T.vectorized(4):
                                with T.block("B_local_local_shared"):
                                    v0 = T.axis.spatial(16384, k_0 * 32 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) // 32 * 4 + ax0_1)
                                    v1 = T.axis.spatial(16384, j_0 * 128 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) % 32 * 4 + ax1_1)
                                    T.reads(B_local_local[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4])
                                    T.writes(B_local_local_shared[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4])
                                    B_local_local_shared[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4] = B_local_local[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4]
                for i_1_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for j_1_0 in T.thread_binding(16, thread="threadIdx.x"):
                        for i_1_1, j_1_1, k_1_0 in T.grid(8, 8, 8):
                            for ax0 in T.vectorized(4):
                                with T.block("A_shared_local"):
                                    v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 8 + i_1_1)
                                    v1 = T.axis.spatial(16384, k_0 * 32 + k_1_0 * 4 + ax0)
                                    T.reads(A_shared[v0, v1])
                                    T.writes(A_shared_local[v0, v1])
                                    A_shared_local[v0, v1] = A_shared[v0, v1]
                            for ax0 in T.vectorized(4):
                                with T.block("B_local_local_shared_local"):
                                    v0 = T.axis.spatial(16384, k_0 * 32 + k_1_0 * 4 + ax0)
                                    v1 = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 8 + j_1_1)
                                    T.reads(B_local_local_shared[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4])
                                    T.writes(B_local_local_shared_local[v1, v0])
                                    B_local_local_shared_local[v1, v0] = B_local_local_shared[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4]
                            with T.block("B_update_o"):
                                vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 8 + i_1_1)
                                vj = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 8 + j_1_1)
                                vk_o = T.axis.reduce(4096, k_0 * 8 + k_1_0)
                                T.reads(C_local[vi, vj], A_shared_local[vi, vk_o * 4 : vk_o * 4 + 4], B_local_local_shared_local[vj, vk_o * 4 : vk_o * 4 + 4])
                                T.writes(C_local[vi, vj])
                                A_1 = T.match_buffer(A_shared_local[vi, vk_o * 4 : vk_o * 4 + 4], [4], dtype="int8", scope="local", align=4, offset_factor=1)
                                B_1 = T.match_buffer(B_local_local_shared_local[vj, vk_o * 4 : vk_o * 4 + 4], [4], dtype="int8", scope="local", align=4, offset_factor=1)
                                C_1 = T.match_buffer(C_local[vi, vj], [1], dtype="int32", scope="local", align=4, offset_factor=1)
                                C_1[0] = T.call_pure_extern("__dp4a", A_1[0:4], B_1[0:4], C_1[0], dtype="int32")
                        for ax0, ax1 in T.grid(8, 8):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 8 + ax0)
                                v1 = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 8 + ax1)
                                T.reads(C_local[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_local[v0, v1]
