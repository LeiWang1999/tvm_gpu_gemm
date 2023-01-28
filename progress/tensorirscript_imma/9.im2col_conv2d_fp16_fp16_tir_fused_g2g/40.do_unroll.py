# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1, 56, 56, 64), "float16"], W: T.Buffer[(3, 3, 64, 64), "float16"], Conv: T.Buffer[(1, 3136, 64), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    A_s0 = T.var("int32")
    A_s0_1 = T.var("int32")
    A_s0_2 = T.var("int32")
    A_s0_3 = T.var("int32")
    A_s1 = T.var("int32")
    A_s1_1 = T.var("int32")
    A_s1_2 = T.var("int32")
    A_s1_3 = T.var("int32")
    B_s0 = T.var("int32")
    B_s1 = T.var("int32")
    C_s0 = T.var("int32")
    C_s0_1 = T.var("int32")
    C_s0_2 = T.var("int32")
    C_s0_3 = T.var("int32")
    C_s0_4 = T.var("int32")
    C_s1 = T.var("int32")
    C_s1_1 = T.var("int32")
    C_s1_2 = T.var("int32")
    C_s1_3 = T.var("int32")
    C_s1_4 = T.var("int32")
    # body
    # with T.block("root")
    data_im2colPad_shared = T.alloc_buffer([1, 196, 36, 16, 16], dtype="float16", scope="shared")
    data_im2colPad_shared_wmma_matrix_a = T.alloc_buffer([1, 196, 36, 16, 16], dtype="float16", scope="wmma.matrix_a")
    weight_flattenPad_shared = T.alloc_buffer([36, 4, 16, 16], dtype="float16", scope="shared")
    weight_flattenPad_shared_wmma_matrix_b = T.alloc_buffer([36, 4, 16, 16], dtype="float16", scope="wmma.matrix_b")
    CPad_shared = T.alloc_buffer([1, 3136, 64], dtype="float16", scope="shared")
    CPad_shared_wmma_accumulator = T.alloc_buffer([1, 196, 4, 16, 16], dtype="float16", scope="wmma.accumulator")
    for n in T.serial(1):
        for x_0_0 in T.thread_binding(196, thread="blockIdx.y"):
            for y_0_0 in T.thread_binding(2, thread="blockIdx.x"):
                for x_0_1 in T.thread_binding(1, thread="threadIdx.y"):
                    for y_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                        for x_0_2_init, y_0_2_init in T.grid(1, 2):
                            with T.block("Conv_init_o"):
                                v_n = T.axis.spatial(1, n)
                                v_x_o = T.axis.spatial(196, x_0_0 + x_0_1 + x_0_2_init)
                                v_y_o = T.axis.spatial(4, y_0_0 * 2 + y_0_1 * 2 + y_0_2_init)
                                T.reads()
                                T.writes(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16])
                                C = T.match_buffer(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0, C_s1], scope="wmma.accumulator", offset_factor=16)
                                T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C_s0 // 16 * (C_s0 // 16) + C.elem_offset % C_s0 // 16, T.float32(0), dtype="handle")
                        for k_0_0 in T.serial(36):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(1):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("data_im2colPad_shared"):
                                                    v0 = T.axis.spatial(1, 0)
                                                    v1 = T.axis.spatial(3136, x_0_0 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 16)
                                                    v2 = T.axis.spatial(576, k_0_0 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(A[v0, v2 // 192 + v1 // 56 - 1, v2 % 192 // 64 + v1 % 56 - 1, v2 % 64])
                                                    T.writes(data_im2colPad_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
                                                    data_im2colPad_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16] = T.if_then_else(v1 < 3136 and v2 < 576, T.if_then_else(1 <= 1 * (v1 // 56) + 1 * (v2 // 64 // 3) and 1 * (v1 // 56) + 1 * (v2 // 64 // 3) < 57 and 1 <= 1 * (v1 % 56) + 1 * (v2 // 64 % 3) and 1 * (v1 % 56) + 1 * (v2 // 64 % 3) < 57, A[v0, 1 * (v1 // 56) + 1 * (v2 // 64 // 3) - 1, 1 * (v1 % 56) + 1 * (v2 // 64 % 3) - 1, v2 % 64], T.float16(0), dtype="float16"), T.float16(0), dtype="float16")
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(2):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("weight_flattenPad_shared"):
                                                    v0 = T.axis.spatial(576, k_0_0 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(64, y_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(W[v0 // 192, v0 % 192 // 64, v0 % 64, v1])
                                                    T.writes(weight_flattenPad_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    weight_flattenPad_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = T.if_then_else(v0 < 576 and v1 < 64, W[v0 // 64 // 3, v0 // 64 % 3, v0 % 64, v1], T.float16(0), dtype="float16")
                            for k_0_1 in T.serial(1):
                                for ax0_0, ax1_0 in T.grid(1, 1):
                                    with T.block("data_im2colPad_shared_wmma.matrix_a_o"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1_o = T.axis.spatial(196, x_0_0 + ax0_0)
                                        v2_o = T.axis.spatial(36, k_0_0 + ax1_0)
                                        T.reads(data_im2colPad_shared[v0, v1_o, v2_o, 0 : 16, 0 : 16])
                                        T.writes(data_im2colPad_shared_wmma_matrix_a[v0, v1_o, v2_o, 0 : 16, 0 : 16])
                                        A_1 = T.match_buffer(data_im2colPad_shared[v0, v1_o, v2_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0, A_s1], scope="shared", offset_factor=16)
                                        C_1 = T.match_buffer(data_im2colPad_shared_wmma_matrix_a[v0, v1_o, v2_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_1, C_s1_1], scope="wmma.matrix_a", offset_factor=16)
                                        T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_s0_1 // 16 * (C_s0_1 // 16) + C_1.elem_offset % C_s0_1 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_1.data, A_1.elem_offset, A_s0 * 16, 1, dtype="handle"), A_s0, "row_major", dtype="handle")
                                for ax0_0, ax1_0 in T.grid(1, 2):
                                    with T.block("weight_flattenPad_shared_wmma.matrix_b_o"):
                                        v0_o = T.axis.spatial(36, k_0_0 + ax0_0)
                                        v1_o = T.axis.spatial(4, y_0_0 * 2 + ax1_0)
                                        T.reads(weight_flattenPad_shared[v0_o, v1_o, 0 : 16, 0 : 16])
                                        T.writes(weight_flattenPad_shared_wmma_matrix_b[v0_o, v1_o, 0 : 16, 0 : 16])
                                        A_2 = T.match_buffer(weight_flattenPad_shared[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_1, A_s1_1], scope="shared", offset_factor=16)
                                        C_2 = T.match_buffer(weight_flattenPad_shared_wmma_matrix_b[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_2, C_s1_2], scope="wmma.matrix_b", offset_factor=16)
                                        T.tvm_load_matrix_sync(C_2.data, 16, 16, 16, C_2.elem_offset // C_s0_2 // 16 * (C_s0_2 // 16) + C_2.elem_offset % C_s0_2 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_2.data, A_2.elem_offset, A_s0_1 * 16, 1, dtype="handle"), A_s0_1, "row_major", dtype="handle")
                                for x_0_2, y_0_2 in T.grid(1, 2):
                                    with T.block("Conv_update_o"):
                                        v_n = T.axis.spatial(1, n)
                                        v_x_o = T.axis.spatial(196, x_0_0 + x_0_1 + x_0_2)
                                        v_y_o = T.axis.spatial(4, y_0_0 * 2 + y_0_1 * 2 + y_0_2)
                                        v_k_o = T.axis.reduce(36, k_0_0 + k_0_1)
                                        T.reads(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16], data_im2colPad_shared_wmma_matrix_a[v_n, v_x_o, v_k_o, 0 : 16, 0 : 16], weight_flattenPad_shared_wmma_matrix_b[v_k_o, v_y_o, 0 : 16, 0 : 16])
                                        T.writes(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16])
                                        A_3 = T.match_buffer(data_im2colPad_shared_wmma_matrix_a[v_n, v_x_o, v_k_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_2, A_s1_2], scope="wmma.matrix_a", offset_factor=16)
                                        B = T.match_buffer(weight_flattenPad_shared_wmma_matrix_b[v_k_o, v_y_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[B_s0, B_s1], scope="wmma.matrix_b", offset_factor=16)
                                        C_3 = T.match_buffer(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_3, C_s1_3], scope="wmma.accumulator", offset_factor=16)
                                        T.tvm_mma_sync(C_3.data, C_3.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16) + C_3.elem_offset % C_s0_3 // 16, A_3.data, A_3.elem_offset // A_s0_2 // 16 * (A_s0_2 // 16) + A_3.elem_offset % A_s0_2 // 16, B.data, B.elem_offset // B_s0 // 16 * (B_s0 // 16) + B.elem_offset % B_s0 // 16, C_3.data, C_3.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16) + C_3.elem_offset % C_s0_3 // 16, dtype="handle")
                        for ax0_0 in T.serial(1):
                            for ax1_0 in T.serial(2):
                                with T.block("CPad_shared_wmma.accumulator_o"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(196, x_0_0 + ax0_0)
                                    v2_o = T.axis.spatial(4, y_0_0 * 2 + ax1_0)
                                    T.reads(CPad_shared_wmma_accumulator[v0, v1_o, v2_o, 0 : 16, 0 : 16])
                                    T.writes(CPad_shared[v0, v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16])
                                    A_4 = T.match_buffer(CPad_shared_wmma_accumulator[v0, v1_o, v2_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_3, A_s1_3], scope="wmma.accumulator", offset_factor=16)
                                    C_4 = T.match_buffer(CPad_shared[v0, v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], [16, 16], dtype="float16", strides=[C_s0_4, C_s1_4], scope="shared", offset_factor=16)
                                    T.tvm_store_matrix_sync(A_4.data, 16, 16, 16, A_4.elem_offset // A_s0_3 // 16 * (A_s0_3 // 16) + A_4.elem_offset % A_s0_3 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_4.data, C_4.elem_offset, C_s0_4 * 16, 2, dtype="handle"), C_s0_4, "row_major", dtype="handle")
                            for ax0_ax1_fused_0 in T.serial(16):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_ax1_fused_2 in T.thread_binding(1, thread="threadIdx.y"):
                                        for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            with T.block("CPad_shared"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(3136, x_0_0 * 16 + (ax0_ax1_fused_0 * 32 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2 * 32 + ax0_ax1_fused_3) // 32)
                                                v2 = T.axis.spatial(64, y_0_0 * 32 + (ax0_ax1_fused_0 * 32 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2 * 32 + ax0_ax1_fused_3) % 32)
                                                T.reads(CPad_shared[v0, v1, v2])
                                                T.writes(Conv[v0, v1, v2])
                                                Conv[v0, v1, v2] = CPad_shared[v0, v1, v2]
