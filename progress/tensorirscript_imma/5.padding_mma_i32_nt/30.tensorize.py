# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # var definition
    tx_4 = T.env_thread("threadIdx.x")
    tx_3 = T.env_thread("threadIdx.x")
    tx_2 = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    tx_1 = T.env_thread("threadIdx.x")
    C_s0 = T.var("int32")
    C_s1 = T.var("int32")
    shared_s0 = T.var("int32")
    shared_s0_1 = T.var("int32")
    shared_s1 = T.var("int32")
    shared_s1_1 = T.var("int32")
    # body
    # with T.block("root")
    PA = T.alloc_buffer([16384], dtype="int32")
    A_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    B_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    A_shared_warp = T.alloc_buffer([1024, 512, 32, 16], dtype="int8", scope="warp")
    B_shared_warp = T.alloc_buffer([1024, 512, 32, 16], dtype="int8", scope="warp")
    C_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="int32", scope="warp")
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
                            C_warp_1 = T.match_buffer(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=1)
                            T.launch_thread(tx, 32)
                            T.mma_fill(8, C_warp_1.data, C_warp_1.elem_offset, dtype="int32")
                    for k_0 in T.serial(256):
                        for ax0_ax1_fused_0 in T.serial(131072):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(16):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(16384, (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 16 + ax0_ax1_fused_3) // 16384)
                                            v1 = T.axis.spatial(16384, (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 16 + ax0_ax1_fused_3) % 16384)
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
                        for i_1_1_0, j_1_1_0, k_1_0 in T.grid(4, 4, 2):
                            with T.block("A_shared_warp_o"):
                                v0_o = T.axis.spatial(1024, i_0 * 8 + i_1_0 * 4 + i_1_1_0)
                                v1_o = T.axis.spatial(512, k_0 * 2 + k_1_0)
                                T.reads(A_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 32 : v1_o * 32 + 32])
                                T.writes(A_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16])
                                warp = T.match_buffer(A_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                shared = T.match_buffer(A_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 32 : v1_o * 32 + 32], [16, 32], dtype="int8", strides=[shared_s0, shared_s1], scope="shared", offset_factor=16)
                                T.launch_thread(tx_1, 32)
                                T.ptx_ldmatrix(False, 4, ".b16", warp.data, warp.elem_offset + 16 * tx_1, T.tvm_access_ptr(T.type_annotation(dtype="int8"), shared.data, shared.elem_offset, shared_s0 * 16, 1, dtype="handle"), shared_s0 * (tx_1 % 16) + 16 * (tx_1 // 16), dtype="int8")
                            with T.block("B_shared_warp_o"):
                                v0_o = T.axis.spatial(1024, j_0 * 16 + j_1_0 * 4 + j_1_1_0)
                                v1_o = T.axis.spatial(512, k_0 * 2 + k_1_0)
                                T.reads(B_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 32 : v1_o * 32 + 32])
                                T.writes(B_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16])
                                warp_1 = T.match_buffer(B_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                shared_1 = T.match_buffer(B_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 32 : v1_o * 32 + 32], [16, 32], dtype="int8", strides=[shared_s0_1, shared_s1_1], scope="shared", offset_factor=16)
                                T.launch_thread(tx_2, 32)
                                T.ptx_ldmatrix(False, 4, ".b16", warp_1.data, warp_1.elem_offset + 16 * tx_2, T.tvm_access_ptr(T.type_annotation(dtype="int8"), shared_1.data, shared_1.elem_offset, shared_s0_1 * 16, 1, dtype="handle"), shared_s0_1 * 8 * (tx_2 // 16) + tx_2 % 8 * shared_s0_1 + 16 * (tx_2 % 16 // 8), dtype="int8")
                            with T.block("B_update_o"):
                                vi_o = T.axis.spatial(1024, i_0 * 8 + i_1_0 * 4 + i_1_1_0)
                                vj_o = T.axis.spatial(1024, j_0 * 16 + j_1_0 * 4 + j_1_1_0)
                                vk_o = T.axis.reduce(512, k_0 * 2 + k_1_0)
                                T.reads(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], A_shared_warp[vi_o, vk_o, 0 : 32, 0 : 16], B_shared_warp[vj_o, vk_o, 0 : 32, 0 : 16])
                                T.writes(C_warp[vi_o, vj_o, 0 : 32, 0 : 8])
                                A_1 = T.match_buffer(A_shared_warp[vi_o, vk_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                B_1 = T.match_buffer(B_shared_warp[vj_o, vk_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                C_1 = T.match_buffer(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=16)
                                T.launch_thread(tx_3, 32)
                                T.ptx_mma("m16n8k32", "row", "col", "int8", "int8", "int32", A_1.data, A_1.elem_offset + tx_3 * 16, B_1.data, B_1.elem_offset + tx_3 * 16, C_1.data, C_1.elem_offset + tx_3 * 8, False, dtype="int32")
                                T.ptx_mma("m16n8k32", "row", "col", "int8", "int8", "int32", A_1.data, A_1.elem_offset + tx_3 * 16, B_1.data, B_1.elem_offset + tx_3 * 16 + T.FloorDiv(16, 2), C_1.data, C_1.elem_offset + tx_3 * 8 + T.FloorDiv(8, 2), False, dtype="int32")
                        for ax0, ax1 in T.grid(16384, 16384):
                            with T.block("Pre_compute_A"):
                                vi, vk = T.axis.remap("SR", [ax0, ax1])
                                T.reads(A_shared[vi, vk])
                                T.writes(PA[vi])
                                with T.init():
                                    PA[vi] = 0
                                PA[vi] = PA[vi] + 1 * T.Cast("int32", A_shared[vi, vk])
                    for ax0_0, ax1_0 in T.grid(4, 4):
                        with T.block("C_warp_o"):
                            v0_o = T.axis.spatial(1024, i_0 * 8 + i_1_0 * 4 + ax0_0)
                            v1_o = T.axis.spatial(1024, j_0 * 16 + j_1_0 * 4 + ax1_0)
                            T.reads(C_warp[v0_o, v1_o, 0 : 32, 0 : 8])
                            T.writes(C[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            C_warp_2 = T.match_buffer(C_warp[v0_o, v1_o, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=1)
                            C_2 = T.match_buffer(C[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="int32", strides=[C_s0, C_s1], offset_factor=1)
                            T.launch_thread(tx_4, 32)
                            T.mma_store(16, 16, T.tvm_access_ptr(T.type_annotation(dtype="int32"), C_2.data, C_2.elem_offset, C_s0 * 16, 2, dtype="handle"), C_warp_2.data, C_warp_2.elem_offset, C_s0, dtype="int32")
