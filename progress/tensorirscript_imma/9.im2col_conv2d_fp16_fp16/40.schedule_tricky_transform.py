# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1, 224, 224, 256), "float16"], W: T.Buffer[(7, 7, 256, 512), "float16"], Conv: T.Buffer[(1, 48400, 512), "float16"]):
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
    CPad = T.alloc_buffer([1, 48400, 512], dtype="float16")
    data_im2col_global = T.alloc_buffer([1, 3025, 784, 16, 16], dtype="float16")
    data_im2col_global_shared = T.alloc_buffer([1, 3025, 784, 16, 16], dtype="float16", scope="shared")
    data_im2col_global_shared_wmma_matrix_a = T.alloc_buffer([1, 3025, 784, 16, 16], dtype="float16", scope="wmma.matrix_a")
    weight_flatten_global = T.alloc_buffer([784, 32, 16, 16], dtype="float16")
    weight_flatten_global_shared = T.alloc_buffer([784, 32, 16, 16], dtype="float16", scope="shared")
    weight_flatten_global_shared_wmma_matrix_b = T.alloc_buffer([784, 32, 16, 16], dtype="float16", scope="wmma.matrix_b")
    CPad_shared = T.alloc_buffer([1, 48400, 512], dtype="float16", scope="shared")
    CPad_shared_wmma_accumulator = T.alloc_buffer([1, 3025, 32, 16, 16], dtype="float16", scope="wmma.accumulator")
    for ax0 in T.serial(1):
        for ax1_ax2_fused_0 in T.thread_binding(4, thread="blockIdx.y"):
            for ax1_ax2_fused_1 in T.thread_binding(2048, thread="blockIdx.x"):
                for ax1_ax2_fused_2 in T.thread_binding(1, thread="vthread.x"):
                    for ax1_ax2_fused_3 in T.thread_binding(128, thread="threadIdx.y"):
                        for ax1_ax2_fused_4 in T.thread_binding(8, thread="threadIdx.x"):
                            for ax1_ax2_fused_5 in T.serial(10):
                                for ax1_ax2_fused_6 in T.vectorized(8):
                                    with T.block("data_im2col_global"):
                                        T.where(((((ax1_ax2_fused_0 * 2048 + ax1_ax2_fused_1 + ax1_ax2_fused_2) * 128 + ax1_ax2_fused_3) * 8 + ax1_ax2_fused_4) * 10 + ax1_ax2_fused_5) * 8 + ax1_ax2_fused_6 < 607129600)
                                        v0 = T.axis.spatial(1, ax0)
                                        v1 = T.axis.spatial(48400, (ax1_ax2_fused_0 * 167772160 + ax1_ax2_fused_1 * 81920 + ax1_ax2_fused_2 * 81920 + ax1_ax2_fused_3 * 640 + ax1_ax2_fused_4 * 80 + ax1_ax2_fused_5 * 8 + ax1_ax2_fused_6) // 12544)
                                        v2 = T.axis.spatial(12544, (ax1_ax2_fused_0 * 167772160 + ax1_ax2_fused_1 * 81920 + ax1_ax2_fused_2 * 81920 + ax1_ax2_fused_3 * 640 + ax1_ax2_fused_4 * 80 + ax1_ax2_fused_5 * 8 + ax1_ax2_fused_6) % 12544)
                                        T.reads(A[v0, v2 // 1792 + v1 // 220 - 1, v2 % 1792 // 256 + v1 % 220 - 1, v2 % 256])
                                        T.writes(data_im2col_global[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
                                        data_im2col_global[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16] = T.if_then_else(1 <= 1 * (v1 // 220) + 1 * (v2 // 256 // 7) and 1 * (v1 // 220) + 1 * (v2 // 256 // 7) < 225 and 1 <= 1 * (v1 % 220) + 1 * (v2 // 256 % 7) and 1 * (v1 % 220) + 1 * (v2 // 256 % 7) < 225, A[v0, 1 * (v1 // 220) + 1 * (v2 // 256 // 7) - 1, 1 * (v1 % 220) + 1 * (v2 // 256 % 7) - 1, v2 % 256], T.float16(0), dtype="float16")
    for ax0_ax1_fused_0 in T.thread_binding(4, thread="blockIdx.y"):
        for ax0_ax1_fused_1 in T.thread_binding(2048, thread="blockIdx.x"):
            for ax0_ax1_fused_2 in T.thread_binding(1, thread="vthread.x"):
                for ax0_ax1_fused_3 in T.thread_binding(128, thread="threadIdx.y"):
                    for ax0_ax1_fused_4 in T.thread_binding(8, thread="threadIdx.x"):
                        for ax0_ax1_fused_5 in T.serial(1):
                            for ax0_ax1_fused_6 in T.vectorized(8):
                                with T.block("weight_flatten_global"):
                                    T.where((((ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 + ax0_ax1_fused_2) * 128 + ax0_ax1_fused_3) * 8 + ax0_ax1_fused_4 + ax0_ax1_fused_5) * 8 + ax0_ax1_fused_6 < 6422528)
                                    v0 = T.axis.spatial(12544, (ax0_ax1_fused_0 * 16777216 + ax0_ax1_fused_1 * 8192 + ax0_ax1_fused_2 * 8192 + ax0_ax1_fused_3 * 64 + ax0_ax1_fused_4 * 8 + ax0_ax1_fused_5 * 8 + ax0_ax1_fused_6) // 512)
                                    v1 = T.axis.spatial(512, (ax0_ax1_fused_0 * 16777216 + ax0_ax1_fused_1 * 8192 + ax0_ax1_fused_2 * 8192 + ax0_ax1_fused_3 * 64 + ax0_ax1_fused_4 * 8 + ax0_ax1_fused_5 * 8 + ax0_ax1_fused_6) % 512)
                                    T.reads(W[v0 // 1792, v0 % 1792 // 256, v0 % 256, v1])
                                    T.writes(weight_flatten_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                    weight_flatten_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = W[v0 // 256 // 7, v0 // 256 % 7, v0 % 256, v1]
    for n in T.serial(1):
        for x_0_0 in T.thread_binding(3025, thread="blockIdx.y"):
            for y_0_0_0 in T.thread_binding(2, thread="blockIdx.z"):
                for y_0_0_1 in T.thread_binding(16, thread="blockIdx.x"):
                    for x_0_1 in T.thread_binding(1, thread="threadIdx.y"):
                        for y_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                            for x_0_2_init, y_0_2_init in T.grid(1, 1):
                                with T.block("Conv_init_o"):
                                    v_n = T.axis.spatial(1, n)
                                    v_x_o = T.axis.spatial(3025, x_0_0 + x_0_1 + x_0_2_init)
                                    v_y_o = T.axis.spatial(32, y_0_0_0 * 16 + y_0_0_1 + y_0_1 + y_0_2_init)
                                    T.reads()
                                    T.writes(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16])
                                    C = T.match_buffer(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0, C_s1], scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C_s0 // 16 * (C_s0 // 16) + C.elem_offset % C_s0 // 16, T.float32(0), dtype="handle")
                            for k_0_0 in T.serial(196):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(4):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                    with T.block("data_im2col_global_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(48400, x_0_0 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                        v2 = T.axis.spatial(12544, k_0_0 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                        T.reads(data_im2col_global[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
                                                        T.writes(data_im2col_global_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
                                                        data_im2col_global_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16] = data_im2col_global[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16]
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(4):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                    with T.block("weight_flatten_global_shared"):
                                                        v0 = T.axis.spatial(12544, k_0_0 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                        v1 = T.axis.spatial(512, y_0_0_0 * 256 + y_0_0_1 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                        T.reads(weight_flatten_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                        T.writes(weight_flatten_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                        weight_flatten_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = weight_flatten_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                                for k_0_1 in T.serial(4):
                                    for ax0_0, ax1_0 in T.grid(1, 1):
                                        with T.block("data_im2col_global_shared_wmma.matrix_a_o"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(3025, x_0_0 + ax0_0)
                                            v2_o = T.axis.spatial(784, k_0_0 * 4 + k_0_1 + ax1_0)
                                            T.reads(data_im2col_global_shared[v0, v1_o, v2_o, 0 : 16, 0 : 16])
                                            T.writes(data_im2col_global_shared_wmma_matrix_a[v0, v1_o, v2_o, 0 : 16, 0 : 16])
                                            A_1 = T.match_buffer(data_im2col_global_shared[v0, v1_o, v2_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0, A_s1], scope="shared", offset_factor=16)
                                            C_1 = T.match_buffer(data_im2col_global_shared_wmma_matrix_a[v0, v1_o, v2_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_1, C_s1_1], scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_s0_1 // 16 * (C_s0_1 // 16) + C_1.elem_offset % C_s0_1 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_1.data, A_1.elem_offset, A_s0 * 16, 1, dtype="handle"), A_s0, "row_major", dtype="handle")
                                    for ax0_0, ax1_0 in T.grid(1, 1):
                                        with T.block("weight_flatten_global_shared_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(784, k_0_0 * 4 + k_0_1 + ax0_0)
                                            v1_o = T.axis.spatial(32, y_0_0_0 * 16 + y_0_0_1 + ax1_0)
                                            T.reads(weight_flatten_global_shared[v0_o, v1_o, 0 : 16, 0 : 16])
                                            T.writes(weight_flatten_global_shared_wmma_matrix_b[v0_o, v1_o, 0 : 16, 0 : 16])
                                            A_2 = T.match_buffer(weight_flatten_global_shared[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_1, A_s1_1], scope="shared", offset_factor=16)
                                            C_2 = T.match_buffer(weight_flatten_global_shared_wmma_matrix_b[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_2, C_s1_2], scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C_2.data, 16, 16, 16, C_2.elem_offset // C_s0_2 // 16 * (C_s0_2 // 16) + C_2.elem_offset % C_s0_2 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_2.data, A_2.elem_offset, A_s0_1 * 16, 1, dtype="handle"), A_s0_1, "row_major", dtype="handle")
                                    for x_0_2, y_0_2 in T.grid(1, 1):
                                        with T.block("Conv_update_o"):
                                            v_n = T.axis.spatial(1, n)
                                            v_x_o = T.axis.spatial(3025, x_0_0 + x_0_1 + x_0_2)
                                            v_y_o = T.axis.spatial(32, y_0_0_0 * 16 + y_0_0_1 + y_0_1 + y_0_2)
                                            v_k_o = T.axis.reduce(784, k_0_0 * 4 + k_0_1)
                                            T.reads(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16], data_im2col_global_shared_wmma_matrix_a[v_n, v_x_o, v_k_o, 0 : 16, 0 : 16], weight_flatten_global_shared_wmma_matrix_b[v_k_o, v_y_o, 0 : 16, 0 : 16])
                                            T.writes(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16])
                                            A_3 = T.match_buffer(data_im2col_global_shared_wmma_matrix_a[v_n, v_x_o, v_k_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_2, A_s1_2], scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(weight_flatten_global_shared_wmma_matrix_b[v_k_o, v_y_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[B_s0, B_s1], scope="wmma.matrix_b", offset_factor=16)
                                            C_3 = T.match_buffer(CPad_shared_wmma_accumulator[v_n, v_x_o, v_y_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_3, C_s1_3], scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C_3.data, C_3.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16) + C_3.elem_offset % C_s0_3 // 16, A_3.data, A_3.elem_offset // A_s0_2 // 16 * (A_s0_2 // 16) + A_3.elem_offset % A_s0_2 // 16, B.data, B.elem_offset // B_s0 // 16 * (B_s0 // 16) + B.elem_offset % B_s0 // 16, C_3.data, C_3.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16) + C_3.elem_offset % C_s0_3 // 16, dtype="handle")
                            for ax0_0, ax1_0 in T.grid(1, 1):
                                with T.block("CPad_shared_wmma.accumulator_o"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(3025, x_0_0 + ax0_0)
                                    v2_o = T.axis.spatial(32, y_0_0_0 * 16 + y_0_0_1 + ax1_0)
                                    T.reads(CPad_shared_wmma_accumulator[v0, v1_o, v2_o, 0 : 16, 0 : 16])
                                    T.writes(CPad_shared[v0, v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16])
                                    A_4 = T.match_buffer(CPad_shared_wmma_accumulator[v0, v1_o, v2_o, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_3, A_s1_3], scope="wmma.accumulator", offset_factor=16)
                                    C_4 = T.match_buffer(CPad_shared[v0, v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], [16, 16], dtype="float16", strides=[C_s0_4, C_s1_4], scope="shared", offset_factor=16)
                                    T.tvm_store_matrix_sync(A_4.data, 16, 16, 16, A_4.elem_offset // A_s0_3 // 16 * (A_s0_3 // 16) + A_4.elem_offset % A_s0_3 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_4.data, C_4.elem_offset, C_s0_4 * 16, 2, dtype="handle"), C_s0_4, "row_major", dtype="handle")
    for ax0, ax1, ax2 in T.grid(1, 48400, 512):
        with T.block("CPad_shared"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(CPad_shared[v0, v1, v2])
            T.writes(CPad[v0, v1, v2])
            CPad[v0, v1, v2] = CPad_shared[v0, v1, v2]
    for n, i, j in T.grid(1, 48400, 512):
        with T.block("CPad"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            T.reads(CPad[vn, vi, vj])
            T.writes(Conv[vn, vi, vj])
            Conv[vn, vi, vj] = CPad[vn, vi, vj]
