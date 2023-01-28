# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(196, 36, 16, 16), "float16"], B: T.Buffer[(36, 4, 16, 16), "float16"], C: T.Buffer[(1, 196, 4, 16, 16), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # var definition
    tx_4 = T.env_thread("threadIdx.x")
    tx_3 = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    tx_2 = T.env_thread("threadIdx.x")
    tx_1 = T.env_thread("threadIdx.x")
    C_s0 = T.var("int32")
    C_s1 = T.var("int32")
    shared_s0 = T.var("int32")
    shared_s0_1 = T.var("int32")
    shared_s1 = T.var("int32")
    shared_s1_1 = T.var("int32")
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([196, 36, 16, 16], dtype="float16", scope="shared")
    A_shared_warp = T.alloc_buffer([196, 36, 32, 8], dtype="float16", scope="warp")
    B_shared = T.alloc_buffer([36, 4, 16, 16], dtype="float16", scope="shared")
    B_shared_warp = T.alloc_buffer([36, 4, 32, 8], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([1, 196, 4, 32, 8], dtype="float16", scope="warp")
    for sk in T.thread_binding(1, thread="blockIdx.z"):
        for ii_0 in T.thread_binding(49, thread="blockIdx.y"):
            for jj_0 in T.thread_binding(1, thread="blockIdx.x"):
                for ii_1 in T.thread_binding(2, thread="threadIdx.y"):
                    for jj_1 in T.thread_binding(2, thread="threadIdx.z"):
                        for ii_2_init, jj_2_init in T.grid(2, 2):
                            with T.block("B_init_o"):
                                vsk = T.axis.spatial(1, sk)
                                vii = T.axis.spatial(196, ii_0 * 4 + ii_1 * 2 + ii_2_init)
                                vjj = T.axis.spatial(4, jj_0 * 4 + jj_1 * 2 + jj_2_init)
                                vi_o = T.axis.spatial(1, 0)
                                vj_o = T.axis.spatial(1, 0)
                                T.reads()
                                T.writes(C_warp[vsk, vii, vjj, 0 : 32, 0 : 8])
                                C_warp_1 = T.match_buffer(C_warp[vsk, vii, vjj, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=1)
                                T.launch_thread(tx, 32)
                                T.mma_fill(8, C_warp_1.data, C_warp_1.elem_offset, dtype="float16")
                        for kk_0 in T.serial(18):
                            for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.serial(2):
                                        for ax0_ax1_ax2_ax3_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_ax3_fused_4 in T.vectorized(8):
                                                with T.block("A_shared"):
                                                    v0 = T.axis.spatial(196, ii_0 * 4 + (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 512 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) // 512)
                                                    v1 = T.axis.spatial(36, kk_0 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 512 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 512 // 256)
                                                    v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 512 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 256 // 16)
                                                    v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 512 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 16)
                                                    T.reads(A[v0, v1, v2 % 8 * 2 + v3 // 8, v2 // 8 * 8 + v3 % 8])
                                                    T.writes(A_shared[v0, v1, v2, v3])
                                                    A_shared[v0, v1, v2, v3] = A[v0, v1, v2 % 8 * 2 + v3 // 8, v2 // 8 * 8 + v3 % 8]
                            for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.serial(2):
                                        for ax0_ax1_ax2_ax3_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_ax3_fused_4 in T.vectorized(8):
                                                with T.block("B_shared"):
                                                    v0 = T.axis.spatial(36, kk_0 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 512 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) // 1024)
                                                    v1 = T.axis.spatial(4, (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 512 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 1024 // 256)
                                                    v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 512 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 256 // 16)
                                                    v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 512 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 16)
                                                    T.reads(B[v0, v1, v2 % 8 * 2 + v3 // 8, v2 // 8 * 8 + v3 % 8])
                                                    T.writes(B_shared[v0, v1, v2, v3])
                                                    B_shared[v0, v1, v2, v3] = B[v0, v1, v2 % 8 * 2 + v3 // 8, v2 // 8 * 8 + v3 % 8]
                            for kk_1 in T.serial(2):
                                for ax0 in T.serial(2):
                                    with T.block("A_shared_warp_o"):
                                        v0 = T.axis.spatial(196, ii_0 * 4 + ii_1 * 2 + ax0)
                                        v1 = T.axis.spatial(36, kk_0 * 2 + kk_1)
                                        v2_o = T.axis.spatial(1, 0)
                                        v3_o = T.axis.spatial(1, 0)
                                        T.reads(A_shared[v0, v1, 0 : 16, 0 : 16])
                                        T.writes(A_shared_warp[v0, v1, 0 : 32, 0 : 8])
                                        warp = T.match_buffer(A_shared_warp[v0, v1, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(A_shared[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[shared_s0, shared_s1], scope="shared", offset_factor=16)
                                        T.launch_thread(tx_1, 32)
                                        T.ptx_ldmatrix(False, 4, ".b16", warp.data, warp.elem_offset + 8 * tx_1, T.tvm_access_ptr(T.type_annotation(dtype="float16"), shared.data, shared.elem_offset, shared_s0 * 16, 1, dtype="handle"), 8 * tx_1, dtype="float16")
                                for ax0 in T.serial(2):
                                    with T.block("B_shared_warp_o"):
                                        v0 = T.axis.spatial(36, kk_0 * 2 + kk_1)
                                        v1 = T.axis.spatial(4, jj_1 * 2 + ax0)
                                        v2_o = T.axis.spatial(1, 0)
                                        v3_o = T.axis.spatial(1, 0)
                                        T.reads(B_shared[v0, v1, 0 : 16, 0 : 16])
                                        T.writes(B_shared_warp[v0, v1, 0 : 32, 0 : 8])
                                        warp_1 = T.match_buffer(B_shared_warp[v0, v1, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        shared_1 = T.match_buffer(B_shared[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[shared_s0_1, shared_s1_1], scope="shared", offset_factor=16)
                                        T.launch_thread(tx_2, 32)
                                        T.ptx_ldmatrix(True, 4, ".b16", warp_1.data, warp_1.elem_offset + 8 * tx_2, T.tvm_access_ptr(T.type_annotation(dtype="float16"), shared_1.data, shared_1.elem_offset, shared_s0_1 * 16, 1, dtype="handle"), 8 * tx_2, dtype="float16")
                                for ii_2, jj_2 in T.grid(2, 2):
                                    with T.block("B_update_o"):
                                        vsk = T.axis.spatial(1, sk)
                                        vii = T.axis.spatial(196, ii_0 * 4 + ii_1 * 2 + ii_2)
                                        vjj = T.axis.spatial(4, jj_0 * 4 + jj_1 * 2 + jj_2)
                                        vkk = T.axis.reduce(36, kk_0 * 2 + kk_1)
                                        vi_o = T.axis.spatial(1, 0)
                                        vj_o = T.axis.spatial(1, 0)
                                        vk_o = T.axis.reduce(1, 0)
                                        T.reads(C_warp[vsk, vii, vjj, 0 : 32, 0 : 8], A_shared_warp[vii, vkk, 0 : 32, 0 : 8], B_shared_warp[vkk, vjj, 0 : 32, 0 : 8])
                                        T.writes(C_warp[vsk, vii, vjj, 0 : 32, 0 : 8])
                                        A_1 = T.match_buffer(A_shared_warp[vii, vkk, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        B_1 = T.match_buffer(B_shared_warp[vkk, vjj, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        C_1 = T.match_buffer(C_warp[vsk, vii, vjj, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        T.launch_thread(tx_3, 32)
                                        T.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx_3 * 8, B_1.data, B_1.elem_offset + tx_3 * 8, C_1.data, C_1.elem_offset + tx_3 * 8, False, dtype="float16")
                                        T.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx_3 * 8, B_1.data, B_1.elem_offset + tx_3 * 8 + T.FloorDiv(8, 2), C_1.data, C_1.elem_offset + tx_3 * 8 + T.FloorDiv(8, 2), False, dtype="float16")
                        for ax0, ax1 in T.grid(2, 2):
                            with T.block("C_warp_o"):
                                v0 = T.axis.spatial(1, 0)
                                v1 = T.axis.spatial(196, ii_0 * 4 + ii_1 * 2 + ax0)
                                v2 = T.axis.spatial(4, jj_1 * 2 + ax1)
                                v3_o = T.axis.spatial(1, 0)
                                v4_o = T.axis.spatial(1, 0)
                                T.reads(C_warp[v0, v1, v2, 0 : 32, 0 : 8])
                                T.writes(C[v0, v1, v2, 0 : 16, 0 : 16])
                                C_warp_2 = T.match_buffer(C_warp[v0, v1, v2, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=1)
                                C_2 = T.match_buffer(C[v0, v1, v2, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0, C_s1], offset_factor=1)
                                T.launch_thread(tx_4, 32)
                                T.mma_store(16, 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_2.data, C_2.elem_offset, C_s0 * 16, 2, dtype="handle"), C_warp_2.data, C_warp_2.elem_offset, C_s0, dtype="float16")
