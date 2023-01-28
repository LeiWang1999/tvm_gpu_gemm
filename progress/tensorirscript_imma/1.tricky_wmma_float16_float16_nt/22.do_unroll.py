# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1024, 1024, 16, 16), "float16"], B: T.Buffer[(1024, 1024, 16, 16), "float16"], C: T.Buffer[(1024, 1024, 16, 16), "float16"]):
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
    A_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="shared")
    A_shared_wmma_matrix_a = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="wmma.matrix_a")
    B_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="shared")
    B_shared_wmma_matrix_b = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="wmma.matrix_b")
    C_wmma_accumulator = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="wmma.accumulator")
    for ii_0 in T.thread_binding(256, thread="blockIdx.y"):
        for jj_0 in T.thread_binding(64, thread="blockIdx.x"):
            for ii_1 in T.thread_binding(2, thread="threadIdx.y"):
                for jj_1 in T.thread_binding(4, thread="threadIdx.z"):
                    for ii_2_init, jj_2_init in T.grid(2, 4):
                        with T.block("B_init_o"):
                            vii = T.axis.spatial(1024, ii_0 * 4 + ii_1 * 2 + ii_2_init)
                            vjj = T.axis.spatial(1024, jj_0 * 16 + jj_1 * 4 + jj_2_init)
                            vi_o = T.axis.spatial(1, 0)
                            vj_o = T.axis.spatial(1, 0)
                            T.reads()
                            T.writes(C_wmma_accumulator[vii, vjj, 0 : 16, 0 : 16])
                            C_1 = T.match_buffer(C_wmma_accumulator[vii, vjj, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0, C_s1], scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_s0 // 16 * (C_s0 // 16) + C_1.elem_offset % C_s0 // 16, T.float32(0), dtype="handle")
                    for kk_0 in T.serial(512):
                        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.serial(1):
                                    for ax0_ax1_ax2_ax3_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_ax3_fused_4 in T.vectorized(8):
                                            with T.block("A_shared"):
                                                v0 = T.axis.spatial(1024, ii_0 * 4 + (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 256 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) // 512)
                                                v1 = T.axis.spatial(1024, kk_0 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 256 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 512 // 256)
                                                v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 256 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 256 // 16)
                                                v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 1024 + ax0_ax1_ax2_ax3_fused_1 * 256 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 16)
                                                T.reads(A[v0, v1, v2, v3])
                                                T.writes(A_shared[v0, v1, v2, v3])
                                                A_shared[v0, v1, v2, v3] = A[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.serial(4):
                                    for ax0_ax1_ax2_ax3_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_ax3_fused_4 in T.vectorized(8):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(1024, jj_0 * 16 + (ax0_ax1_ax2_ax3_fused_0 * 4096 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) // 512)
                                                v1 = T.axis.spatial(1024, kk_0 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 4096 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 512 // 256)
                                                v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 4096 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 256 // 16)
                                                v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 4096 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 256 + ax0_ax1_ax2_ax3_fused_3 * 8 + ax0_ax1_ax2_ax3_fused_4) % 16)
                                                T.reads(B[v0, v1, v2, v3])
                                                T.writes(B_shared[v0, v1, v2, v3])
                                                B_shared[v0, v1, v2, v3] = B[v0, v1, v2, v3]
                        for kk_1 in T.serial(2):
                            for ax0 in T.serial(2):
                                with T.block("A_shared_wmma.matrix_a_o"):
                                    v0 = T.axis.spatial(1024, ii_0 * 4 + ii_1 * 2 + ax0)
                                    v1 = T.axis.spatial(1024, kk_0 * 2 + kk_1)
                                    v2_o = T.axis.spatial(1, 0)
                                    v3_o = T.axis.spatial(1, 0)
                                    T.reads(A_shared[v0, v1, 0 : 16, 0 : 16])
                                    T.writes(A_shared_wmma_matrix_a[v0, v1, 0 : 16, 0 : 16])
                                    A_1 = T.match_buffer(A_shared[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0, A_s1], scope="shared", offset_factor=16)
                                    C_2 = T.match_buffer(A_shared_wmma_matrix_a[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_1, C_s1_1], scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_2.data, 16, 16, 16, C_2.elem_offset // C_s0_1 // 16 * (C_s0_1 // 16) + C_2.elem_offset % C_s0_1 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_1.data, A_1.elem_offset, A_s0 * 16, 1, dtype="handle"), A_s0, "row_major", dtype="handle")
                            for ax0 in T.serial(4):
                                with T.block("B_shared_wmma.matrix_b_o"):
                                    v0 = T.axis.spatial(1024, jj_0 * 16 + jj_1 * 4 + ax0)
                                    v1 = T.axis.spatial(1024, kk_0 * 2 + kk_1)
                                    v2_o = T.axis.spatial(1, 0)
                                    v3_o = T.axis.spatial(1, 0)
                                    T.reads(B_shared[v0, v1, 0 : 16, 0 : 16])
                                    T.writes(B_shared_wmma_matrix_b[v0, v1, 0 : 16, 0 : 16])
                                    A_2 = T.match_buffer(B_shared[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_1, A_s1_1], scope="shared", offset_factor=16)
                                    C_3 = T.match_buffer(B_shared_wmma_matrix_b[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_2, C_s1_2], scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_3.data, 16, 16, 16, C_3.elem_offset // C_s0_2 // 16 * (C_s0_2 // 16) + C_3.elem_offset % C_s0_2 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_2.data, A_2.elem_offset, A_s0_1 * 16, 1, dtype="handle"), A_s0_1, "col_major", dtype="handle")
                            for ii_2, jj_2 in T.grid(2, 4):
                                with T.block("B_update_o"):
                                    vii = T.axis.spatial(1024, ii_0 * 4 + ii_1 * 2 + ii_2)
                                    vjj = T.axis.spatial(1024, jj_0 * 16 + jj_1 * 4 + jj_2)
                                    vkk = T.axis.reduce(1024, kk_0 * 2 + kk_1)
                                    vi_o = T.axis.spatial(1, 0)
                                    vj_o = T.axis.spatial(1, 0)
                                    vk_o = T.axis.reduce(1, 0)
                                    T.reads(C_wmma_accumulator[vii, vjj, 0 : 16, 0 : 16], A_shared_wmma_matrix_a[vii, vkk, 0 : 16, 0 : 16], B_shared_wmma_matrix_b[vjj, vkk, 0 : 16, 0 : 16])
                                    T.writes(C_wmma_accumulator[vii, vjj, 0 : 16, 0 : 16])
                                    A_3 = T.match_buffer(A_shared_wmma_matrix_a[vii, vkk, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_2, A_s1_2], scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_shared_wmma_matrix_b[vjj, vkk, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[B_s0, B_s1], scope="wmma.matrix_b", offset_factor=16)
                                    C_4 = T.match_buffer(C_wmma_accumulator[vii, vjj, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_3, C_s1_3], scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_4.data, C_4.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16) + C_4.elem_offset % C_s0_3 // 16, A_3.data, A_3.elem_offset // A_s0_2 // 16 * (A_s0_2 // 16) + A_3.elem_offset % A_s0_2 // 16, B_1.data, B_1.elem_offset // B_s0 // 16 * (B_s0 // 16) + B_1.elem_offset % B_s0 // 16, C_4.data, C_4.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16) + C_4.elem_offset % C_s0_3 // 16, dtype="handle")
                    for ax0, ax1 in T.grid(2, 4):
                        with T.block("C_wmma.accumulator_o"):
                            v0 = T.axis.spatial(1024, ii_0 * 4 + ii_1 * 2 + ax0)
                            v1 = T.axis.spatial(1024, jj_0 * 16 + jj_1 * 4 + ax1)
                            v2_o = T.axis.spatial(1, 0)
                            v3_o = T.axis.spatial(1, 0)
                            T.reads(C_wmma_accumulator[v0, v1, 0 : 16, 0 : 16])
                            T.writes(C[v0, v1, 0 : 16, 0 : 16])
                            A_4 = T.match_buffer(C_wmma_accumulator[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[A_s0_3, A_s1_3], scope="wmma.accumulator", offset_factor=16)
                            C_5 = T.match_buffer(C[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="float16", strides=[C_s0_4, C_s1_4], offset_factor=16)
                            T.tvm_store_matrix_sync(A_4.data, 16, 16, 16, A_4.elem_offset // A_s0_3 // 16 * (A_s0_3 // 16) + A_4.elem_offset % A_s0_3 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_5.data, C_5.elem_offset, C_s0_4 * 16, 2, dtype="handle"), C_s0_4, "row_major", dtype="handle")
