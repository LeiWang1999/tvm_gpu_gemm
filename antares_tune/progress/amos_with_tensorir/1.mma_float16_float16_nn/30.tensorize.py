# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(8192, 8192), "float16"], B: T.Buffer[(8192, 8192), "float16"], C: T.Buffer[(8192, 8192), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # var definition
    tx_3 = T.env_thread("threadIdx.x")
    tx_2 = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    tx_4 = T.env_thread("threadIdx.x")
    tx_1 = T.env_thread("threadIdx.x")
    C_s0 = T.var("int32")
    C_s1 = T.var("int32")
    shared_s0 = T.var("int32")
    shared_s0_1 = T.var("int32")
    shared_s1 = T.var("int32")
    shared_s1_1 = T.var("int32")
    # body
    # with T.block("root")
    A_global = T.alloc_buffer([512, 512, 16, 16], dtype="float16")
    A_global_shared = T.alloc_buffer([512, 512, 16, 16], dtype="float16", scope="shared")
    A_global_shared_warp = T.alloc_buffer([512, 512, 32, 8], dtype="float16", scope="warp")
    B_global = T.alloc_buffer([512, 512, 16, 16], dtype="float16")
    B_global_shared = T.alloc_buffer([512, 512, 16, 16], dtype="float16", scope="shared")
    B_global_shared_warp = T.alloc_buffer([512, 512, 32, 8], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([512, 512, 32, 8], dtype="float16", scope="warp")
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("B_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8])
            B_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8] = B[v0, v1]
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8])
            A_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8] = A[v0, v1]
    for i_0_0 in T.thread_binding(64, thread="blockIdx.y"):
        for j_0_0_0 in T.thread_binding(2, thread="blockIdx.z"):
            for j_0_0_1 in T.thread_binding(16, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(1, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(4, thread="threadIdx.z"):
                        for i_0_2_init, j_0_2_init in T.grid(8, 4):
                            with T.block("B_init_o"):
                                vi_o = T.axis.spatial(512, i_0_0 * 8 + i_0_1 * 8 + i_0_2_init)
                                vj_o = T.axis.spatial(512, j_0_0_0 * 256 + j_0_0_1 * 16 + j_0_1 * 4 + j_0_2_init)
                                T.reads()
                                T.writes(C_warp[vi_o, vj_o, 0 : 32, 0 : 8])
                                C_warp_1 = T.match_buffer(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=1)
                                T.launch_thread(tx, 32)
                                T.mma_fill(8, C_warp_1.data, C_warp_1.elem_offset, dtype="float16")
                        for k_0_0 in T.serial(256):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(4):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("A_global_shared"):
                                                    v0 = T.axis.spatial(8192, i_0_0 * 128 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 512 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(8192, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 512 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(A_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8])
                                                    T.writes(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8]
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(8):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("B_global_shared"):
                                                    v0 = T.axis.spatial(8192, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 4096 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(8192, j_0_0_0 * 4096 + j_0_0_1 * 256 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 4096 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(B_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8])
                                                    T.writes(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8]
                            for k_0_1 in T.serial(2):
                                for ax0_0, ax1_0 in T.grid(8, 1):
                                    with T.block("A_global_shared_warp_o"):
                                        v0_o = T.axis.spatial(512, i_0_0 * 8 + ax0_0)
                                        v1_o = T.axis.spatial(512, k_0_0 * 2 + k_0_1 + ax1_0)
                                        T.reads(A_global_shared[v0_o, v1_o, 0 : 16, 0 : 16])
                                        T.writes(A_global_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8])
                                        warp = T.match_buffer(A_global_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(A_global_shared[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[shared_s0, shared_s1], scope="shared", offset_factor=16)
                                        T.launch_thread(tx_1, 32)
                                        T.ptx_ldmatrix(False, 4, ".b16", warp.data, warp.elem_offset + 8 * tx_1, T.tvm_access_ptr(T.type_annotation(dtype="float16"), shared.data, shared.elem_offset, shared_s0 * 16, 1, dtype="handle"), 8 * tx_1, dtype="float16")
                                for ax0_0, ax1_0 in T.grid(1, 4):
                                    with T.block("B_global_shared_warp_o"):
                                        v0_o = T.axis.spatial(512, k_0_0 * 2 + k_0_1 + ax0_0)
                                        v1_o = T.axis.spatial(512, j_0_0_0 * 256 + j_0_0_1 * 16 + j_0_1 * 4 + ax1_0)
                                        T.reads(B_global_shared[v0_o, v1_o, 0 : 16, 0 : 16])
                                        T.writes(B_global_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8])
                                        warp_1 = T.match_buffer(B_global_shared_warp[v0_o, v1_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        shared_1 = T.match_buffer(B_global_shared[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[shared_s0_1, shared_s1_1], scope="shared", offset_factor=16)
                                        T.launch_thread(tx_2, 32)
                                        T.ptx_ldmatrix(True, 4, ".b16", warp_1.data, warp_1.elem_offset + 8 * tx_2, T.tvm_access_ptr(T.type_annotation(dtype="float16"), shared_1.data, shared_1.elem_offset, shared_s0_1 * 16, 1, dtype="handle"), 8 * tx_2, dtype="float16")
                                for i_0_2, j_0_2 in T.grid(8, 4):
                                    with T.block("B_update_o"):
                                        vi_o = T.axis.spatial(512, i_0_0 * 8 + i_0_1 * 8 + i_0_2)
                                        vj_o = T.axis.spatial(512, j_0_0_0 * 256 + j_0_0_1 * 16 + j_0_1 * 4 + j_0_2)
                                        vk_o = T.axis.reduce(512, k_0_0 * 2 + k_0_1)
                                        T.reads(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], A_global_shared_warp[vi_o, vk_o, 0 : 32, 0 : 8], B_global_shared_warp[vk_o, vj_o, 0 : 32, 0 : 8])
                                        T.writes(C_warp[vi_o, vj_o, 0 : 32, 0 : 8])
                                        A_1 = T.match_buffer(A_global_shared_warp[vi_o, vk_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        B_1 = T.match_buffer(B_global_shared_warp[vk_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        C_1 = T.match_buffer(C_warp[vi_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=16)
                                        T.launch_thread(tx_3, 32)
                                        T.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx_3 * 8, B_1.data, B_1.elem_offset + tx_3 * 8, C_1.data, C_1.elem_offset + tx_3 * 8, False, dtype="float16")
                                        T.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx_3 * 8, B_1.data, B_1.elem_offset + tx_3 * 8 + T.FloorDiv(8, 2), C_1.data, C_1.elem_offset + tx_3 * 8 + T.FloorDiv(8, 2), False, dtype="float16")
                        for ax0_0, ax1_0 in T.grid(8, 4):
                            with T.block("C_warp_o"):
                                v0_o = T.axis.spatial(512, i_0_0 * 8 + ax0_0)
                                v1_o = T.axis.spatial(512, j_0_0_0 * 256 + j_0_0_1 * 16 + j_0_1 * 4 + ax1_0)
                                T.reads(C_warp[v0_o, v1_o, 0 : 32, 0 : 8])
                                T.writes(C[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                C_warp_2 = T.match_buffer(C_warp[v0_o, v1_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=1)
                                C_2 = T.match_buffer(C[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[C_s0, C_s1], offset_factor=1)
                                T.launch_thread(tx_4, 32)
                                T.mma_store(16, 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_2.data, C_2.elem_offset, C_s0 * 16, 2, dtype="handle"), C_warp_2.data, C_warp_2.elem_offset, C_s0, dtype="float16")
