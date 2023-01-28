# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "float16"], B: T.Buffer[(16384, 16384), "float16"], C: T.Buffer[(16384, 16384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # var definition
    tx_3 = T.env_thread("threadIdx.x")
    tx_2 = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    tx_1 = T.env_thread("threadIdx.x")
    shared_s0 = T.var("int32")
    shared_s0_1 = T.var("int32")
    shared_s1 = T.var("int32")
    shared_s1_1 = T.var("int32")
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([16384, 16384], dtype="float16", scope="shared")
    B_shared = T.alloc_buffer([16384, 16384], dtype="float16", scope="shared")
    A_shared_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    B_shared_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="float16", scope="warp")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for i_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                for j_1_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for i_1_1_0_init, j_1_1_0_init in T.grid(4, 4):
                        with T.block("B_init_o"):
                            vi_o = T.axis.spatial(1024, i_0 * 8 + i_1_0 * 4 + i_1_1_0_init)
                            vj_o = T.axis.spatial(1024, j_0 * 16 + j_1_0 * 4 + j_1_1_0_init)
                            T.reads()
                            T.writes(C_warp[vi_o, vj_o, 0 : 32, 0 : 8])
                            C_warp_1 = T.match_buffer(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=1)
                            T.launch_thread(tx, 32)
                            T.mma_fill(8, C_warp_1.data, C_warp_1.elem_offset, dtype="float16")
                    for k_0 in T.serial(512):
                        for ax0_ax1_fused_0 in T.serial(2):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_ax1_fused_2 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_4 in T.vectorized(8):
                                            with T.block("A_shared"):
                                                v0 = T.axis.spatial(16384, i_0 * 128 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 32)
                                                v1 = T.axis.spatial(16384, k_0 * 32 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 32)
                                                T.reads(A[v0, v1])
                                                T.writes(A_shared[v0, v1])
                                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                                A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(4):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_ax1_fused_2 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_4 in T.vectorized(8):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(16384, j_0 * 256 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 32)
                                                v1 = T.axis.spatial(16384, k_0 * 32 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 32)
                                                T.reads(B[v0, v1])
                                                T.writes(B_shared[v0, v1])
                                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                                B_shared[v0, v1] = B[v0, v1]
                        for i_1_1_0, j_1_1_0, k_1_0 in T.grid(4, 4, 2):
                            with T.block("A_shared_warp_o"):
                                v0_o = T.axis.spatial(1024, i_0 * 8 + i_1_0 * 4 + i_1_1_0)
                                v1_o = T.axis.spatial(1024, k_0 * 2 + k_1_0)
                                T.reads(A_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                T.writes(A_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8])
                                warp = T.match_buffer(A_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                shared = T.match_buffer(A_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[shared_s0, shared_s1], scope="shared", offset_factor=16)
                                T.launch_thread(tx_1, 32)
                                T.ptx_ldmatrix(False, 4, ".b16", warp.data, warp.elem_offset + 8 * tx_1, T.tvm_access_ptr(T.type_annotation(dtype="float16"), shared.data, shared.elem_offset, shared_s0 * 16, 1, dtype="handle"), shared_s0 * (tx_1 % 16) + 8 * (tx_1 // 16), dtype="float16")
                            with T.block("B_shared_warp_o"):
                                v0_o = T.axis.spatial(1024, j_0 * 16 + j_1_0 * 4 + j_1_1_0)
                                v1_o = T.axis.spatial(1024, k_0 * 2 + k_1_0)
                                T.reads(B_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                T.writes(B_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8])
                                warp_1 = T.match_buffer(B_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                shared_1 = T.match_buffer(B_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[shared_s0_1, shared_s1_1], scope="shared", offset_factor=16)
                                T.launch_thread(tx_2, 32)
                                T.ptx_ldmatrix(False, 4, ".b16", warp_1.data, warp_1.elem_offset + 8 * tx_2, T.tvm_access_ptr(T.type_annotation(dtype="float16"), shared_1.data, shared_1.elem_offset, shared_s0_1 * 16, 1, dtype="handle"), shared_s0_1 * 8 * (tx_2 // 16) + shared_s0_1 * (tx_2 % 8) + 8 * (tx_2 % 16 // 8), dtype="float16")
                            with T.block("B_update_o"):
                                vi_o = T.axis.spatial(1024, i_0 * 8 + i_1_0 * 4 + i_1_1_0)
                                vj_o = T.axis.spatial(1024, j_0 * 16 + j_1_0 * 4 + j_1_1_0)
                                vk_o = T.axis.reduce(1024, k_0 * 2 + k_1_0)
                                T.reads(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], A_shared_warp[vi_o, vk_o, 0 : 32, 0 : 8], B_shared_warp[vj_o, vk_o, 0 : 32, 0 : 8])
                                T.writes(C_warp[vi_o, vj_o, 0 : 32, 0 : 8])
                                A_1 = T.match_buffer(A_shared_warp[vi_o, vk_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                B_1 = T.match_buffer(B_shared_warp[vj_o, vk_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                C_1 = T.match_buffer(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                T.launch_thread(tx_3, 32)
                                T.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx_3 * 8, B_1.data, B_1.elem_offset + tx_3 * 8, C_1.data, C_1.elem_offset + tx_3 * 8, False, dtype="float16")
                                T.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx_3 * 8, B_1.data, B_1.elem_offset + tx_3 * 8 + T.FloorDiv(8, 2), C_1.data, C_1.elem_offset + tx_3 * 8 + T.FloorDiv(8, 2), False, dtype="float16")
                    for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 4, 16, 16):
                        with T.block("C_warp"):
                            v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 64 + ax0_0 * 16 + ax0_1)
                            v1 = T.axis.spatial(16384, j_0 * 256 + j_1_0 * 64 + ax1_0 * 16 + ax1_1)
                            T.reads(C_warp[v0 // 16, v1 // 16, v0 % 8 * 4 + v1 % 8 // 2, v1 % 16 // 8 * 4 + v0 % 16 // 8 * 2 + v1 % 2])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_warp[v0 // 16, v1 // 16, v0 % 8 * 4 + v1 % 8 // 2, v1 % 16 // 8 * 4 + v0 % 16 // 8 * 2 + v1 % 2]
