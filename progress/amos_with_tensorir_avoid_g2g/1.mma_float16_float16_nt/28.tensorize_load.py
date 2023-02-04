# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "float16"], B: T.Buffer[(16384, 16384), "float16"], C: T.Buffer[(16384, 16384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # var definition
    tx = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    shared_s0 = T.var("int32")
    shared_s1 = T.var("int32")
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="shared")
    A_shared_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    B_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="shared")
    B_shared_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    for i_0_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0_0_0 in T.thread_binding(4, thread="blockIdx.z"):
            for j_0_0_1 in T.thread_binding(16, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(4, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                        for i_0_2_init, j_0_2_init in T.grid(2, 16):
                            with T.block("B_init_o"):
                                vi_o = T.axis.spatial(1024, i_0_0 * 8 + i_0_1 * 2 + i_0_2_init)
                                vj_o = T.axis.spatial(1024, j_0_0_0 * 256 + j_0_0_1 * 16 + j_0_2_init)
                                T.reads()
                                T.writes(C_warp[vi_o, vj_o, 0 : 32, 0 : 8])
                                C_warp_1 = T.match_buffer(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=1)
                                T.launch_thread(tx, 32)
                                T.mma_fill(8, C_warp_1.data, C_warp_1.elem_offset, dtype="float16")
                        for k_0_0 in T.serial(512):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(4):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("A_shared"):
                                                    v0 = T.axis.spatial(16384, i_0_0 * 128 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 512 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(16384, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 512 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(A[v0 // 16 * 16 + v0 % 8 * 2 + v1 % 16 // 8, v1 // 16 * 16 + v0 % 16 // 8 * 8 + v1 % 8])
                                                    T.writes(A_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    A_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A[v0 // 16 * 16 + v0 % 8 * 2 + v1 % 16 // 8, v1 // 16 * 16 + v0 % 16 // 8 * 8 + v1 % 8]
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(8):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("B_shared"):
                                                    v0 = T.axis.spatial(16384, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 4096 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(16384, j_0_0_0 * 4096 + j_0_0_1 * 256 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 4096 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(B[v0 // 16 * 16 + v0 % 8 * 2 + v1 % 16 // 8, v1 // 16 * 16 + v0 % 16 // 8 * 8 + v1 % 8])
                                                    T.writes(B_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    B_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B[v0 // 16 * 16 + v0 % 8 * 2 + v1 % 16 // 8, v1 // 16 * 16 + v0 % 16 // 8 * 8 + v1 % 8]
                            for k_0_1 in T.serial(2):
                                for ax0_0, ax1_0 in T.grid(2, 1):
                                    with T.block("A_shared_warp_o"):
                                        v0_o = T.axis.spatial(1024, i_0_0 * 8 + i_0_1 * 2 + ax0_0)
                                        v1_o = T.axis.spatial(1024, k_0_0 * 2 + k_0_1)
                                        T.reads(A_shared[v0_o, v1_o, 0 : 16, 0 : 16])
                                        T.writes(A_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8])
                                        warp = T.match_buffer(A_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(A_shared[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[shared_s0, shared_s1], scope="shared", offset_factor=16)
                                        T.launch_thread(tx, 32)
                                        T.ptx_ldmatrix(False, 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation(dtype="float16"), shared.data, shared.elem_offset, shared_s0 * 16, 1, dtype="handle"), 8 * tx, dtype="float16")
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(1, 16, 16, 16):
                                    with T.block("B_shared_warp"):
                                        v0 = T.axis.spatial(16384, k_0_0 * 32 + k_0_1 * 16 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(16384, j_0_0_0 * 4096 + j_0_0_1 * 256 + ax1_0 * 16 + ax1_1)
                                        T.reads(B_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        T.writes(B_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8])
                                        B_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8] = B_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                                for i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(2, 16, 16, 16, 16):
                                    with T.block("B_update"):
                                        vi = T.axis.spatial(16384, i_0_0 * 128 + i_0_1 * 32 + i_0_2 * 16 + i_1)
                                        vj = T.axis.spatial(16384, j_0_0_0 * 4096 + j_0_0_1 * 256 + j_0_1 * 256 + j_0_2 * 16 + j_1)
                                        vk = T.axis.reduce(16384, k_0_0 * 32 + k_0_1 * 16 + k_1)
                                        T.reads(C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2], A_shared_warp[vi // 16, vk // 16, vi % 16 * 2 + vk % 16 // 8, vk % 8], B_shared_warp[vk // 16, vj // 16, vk % 16 * 2 + vj % 16 // 8, vj % 8])
                                        T.writes(C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2])
                                        C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2] = C_warp[vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2] + A_shared_warp[vi // 16, vk // 16, vi % 16 * 2 + vk % 16 // 8, vk % 8] * B_shared_warp[vk // 16, vj // 16, vk % 16 * 2 + vj % 16 // 8, vj % 8]
                        for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(2, 16, 16, 16):
                            with T.block("C_warp"):
                                v0 = T.axis.spatial(16384, i_0_0 * 128 + i_0_1 * 32 + ax0_0 * 16 + ax0_1)
                                v1 = T.axis.spatial(16384, j_0_0_0 * 4096 + j_0_0_1 * 256 + ax1_0 * 16 + ax1_1)
                                T.reads(C_warp[v0 // 16, v1 // 16, v0 % 8 * 4 + v1 % 8 // 2, v1 % 16 // 8 * 4 + v0 % 16 // 8 * 2 + v1 % 2])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_warp[v0 // 16, v1 // 16, v0 % 8 * 4 + v1 % 8 // 2, v1 % 16 // 8 * 4 + v0 % 16 // 8 * 2 + v1 % 2]