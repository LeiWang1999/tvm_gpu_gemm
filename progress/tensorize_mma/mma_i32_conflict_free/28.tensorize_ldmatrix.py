# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # var definition
    tx = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    shared_s0 = T.var("int32")
    shared_s0_1 = T.var("int32")
    shared_s1 = T.var("int32")
    shared_s1_1 = T.var("int32")
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
                            for ax0_0, ax1_0 in T.grid(2, 2):
                                with T.block("A_shared_warp_o"):
                                    v0_o = T.axis.spatial(512, i_0 * 4 + i_1_0 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(1024, k_0 * 4 + k_1_0 * 2 + ax1_0)
                                    T.reads(A_shared[v0_o * 32 : v0_o * 32 + 32, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(A_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16])
                                    warp = T.match_buffer(A_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                    shared = T.match_buffer(A_shared[v0_o * 32 : v0_o * 32 + 32, v1_o * 16 : v1_o * 16 + 16], [32, 16], dtype="int8", strides=[shared_s0, shared_s1], scope="shared", offset_factor=16)
                                    T.launch_thread(tx, 32)
                                    T.evaluate(T.ptx_ldmatrix(True, 4, ".b16", warp.data, warp.elem_offset + 16 * tx, T.tvm_access_ptr(T.type_annotation(dtype="int8"), shared.data, shared.elem_offset, shared_s0 * 32, 1, dtype="handle"), shared_s0, dtype="int8"))
                            for ax0_0, ax1_0 in T.grid(2, 2):
                                with T.block("B_shared_warp_o"):
                                    v0_o = T.axis.spatial(512, j_0 * 8 + j_1_0 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(1024, k_0 * 4 + k_1_0 * 2 + ax1_0)
                                    T.reads(B_shared[v0_o * 32 : v0_o * 32 + 32, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(B_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16])
                                    warp_1 = T.match_buffer(B_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                    shared_1 = T.match_buffer(B_shared[v0_o * 32 : v0_o * 32 + 32, v1_o * 16 : v1_o * 16 + 16], [32, 16], dtype="int8", strides=[shared_s0_1, shared_s1_1], scope="shared", offset_factor=16)
                                    T.launch_thread(tx, 32)
                                    T.evaluate(T.ptx_ldmatrix(True, 4, ".b16", warp_1.data, warp_1.elem_offset + 16 * tx, T.tvm_access_ptr(T.type_annotation(dtype="int8"), shared_1.data, shared_1.elem_offset, shared_s0_1 * 32, 1, dtype="handle"), shared_s0_1, dtype="int8"))
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