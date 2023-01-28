# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
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
    PA = T.alloc_buffer([16384], dtype="int32")
    PB = T.alloc_buffer([16384], dtype="int32")
    QC = T.alloc_buffer([16384, 16384], dtype="int32")
    A_global = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8")
    A_global_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    A_global_shared_wmma_matrix_a = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_a")
    B_global = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8")
    B_global_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    B_global_shared_wmma_matrix_b = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_b")
    QC_wmma_accumulator = T.alloc_buffer([1024, 1024, 16, 16], dtype="int32", scope="wmma.accumulator")
    for ax0_ax1_fused_0 in T.thread_binding(4, thread="blockIdx.y"):
        for ax0_ax1_fused_1 in T.thread_binding(2048, thread="blockIdx.x"):
            for ax0_ax1_fused_2 in T.thread_binding(1, thread="vthread.x"):
                for ax0_ax1_fused_3 in T.thread_binding(128, thread="threadIdx.y"):
                    for ax0_ax1_fused_4 in T.thread_binding(8, thread="threadIdx.x"):
                        for ax0_ax1_fused_5 in T.serial(2):
                            for ax0_ax1_fused_6 in T.vectorized(16):
                                with T.block("B_global"):
                                    v0 = T.axis.spatial(16384, (ax0_ax1_fused_0 * 67108864 + ax0_ax1_fused_1 * 32768 + ax0_ax1_fused_2 * 32768 + ax0_ax1_fused_3 * 256 + ax0_ax1_fused_4 * 32 + ax0_ax1_fused_5 * 16 + ax0_ax1_fused_6) // 16384)
                                    v1 = T.axis.spatial(16384, (ax0_ax1_fused_0 * 67108864 + ax0_ax1_fused_1 * 32768 + ax0_ax1_fused_2 * 32768 + ax0_ax1_fused_3 * 256 + ax0_ax1_fused_4 * 32 + ax0_ax1_fused_5 * 16 + ax0_ax1_fused_6) % 16384)
                                    T.reads(B[v0, v1])
                                    T.writes(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                    B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B[v0, v1]
    for ax0_ax1_fused_0 in T.thread_binding(4, thread="blockIdx.y"):
        for ax0_ax1_fused_1 in T.thread_binding(2048, thread="blockIdx.x"):
            for ax0_ax1_fused_2 in T.thread_binding(1, thread="vthread.x"):
                for ax0_ax1_fused_3 in T.thread_binding(128, thread="threadIdx.y"):
                    for ax0_ax1_fused_4 in T.thread_binding(8, thread="threadIdx.x"):
                        for ax0_ax1_fused_5 in T.serial(2):
                            for ax0_ax1_fused_6 in T.vectorized(16):
                                with T.block("A_global"):
                                    v0 = T.axis.spatial(16384, (ax0_ax1_fused_0 * 67108864 + ax0_ax1_fused_1 * 32768 + ax0_ax1_fused_2 * 32768 + ax0_ax1_fused_3 * 256 + ax0_ax1_fused_4 * 32 + ax0_ax1_fused_5 * 16 + ax0_ax1_fused_6) // 16384)
                                    v1 = T.axis.spatial(16384, (ax0_ax1_fused_0 * 67108864 + ax0_ax1_fused_1 * 32768 + ax0_ax1_fused_2 * 32768 + ax0_ax1_fused_3 * 256 + ax0_ax1_fused_4 * 32 + ax0_ax1_fused_5 * 16 + ax0_ax1_fused_6) % 16384)
                                    T.reads(A[v0, v1])
                                    T.writes(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                    A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A[v0, v1]
    for i_0_0 in T.thread_binding(64, thread="blockIdx.y"):
        for j_0_0 in T.thread_binding(256, thread="blockIdx.x"):
            for i_0_1 in T.thread_binding(4, thread="threadIdx.y"):
                for j_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                    for i_0_2_init, j_0_2_init in T.grid(4, 4):
                        with T.block("B_init_o"):
                            vi_o = T.axis.spatial(1024, i_0_0 * 16 + i_0_1 * 4 + i_0_2_init)
                            vj_o = T.axis.spatial(1024, j_0_0 * 4 + j_0_1 * 4 + j_0_2_init)
                            T.reads()
                            T.writes(QC_wmma_accumulator[vi_o, vj_o, 0 : 16, 0 : 16])
                            C_1 = T.match_buffer(QC_wmma_accumulator[vi_o, vj_o, 0 : 16, 0 : 16], [16, 16], dtype="int32", strides=[C_s0, C_s1], scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_s0 // 16 * (C_s0 // 16) + C_1.elem_offset % C_s0 // 16, T.float32(0), dtype="handle")
                    for k_0_0 in T.serial(256, annotations={"thread_rasterization":32}):
                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(131072):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(16):
                                            with T.block("A_global_shared"):
                                                v0 = T.axis.spatial(16384, (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 67108864 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 67108864 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 262144 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 67108864 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 67108864 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                v1 = T.axis.spatial(16384, (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 67108864 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 67108864 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 262144 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 67108864 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 67108864 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                T.reads(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                T.writes(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(2):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(16):
                                            with T.block("B_global_shared"):
                                                v0 = T.axis.spatial(16384, j_0_0 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 1024 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                v1 = T.axis.spatial(16384, k_0_0 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 1024 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                T.reads(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                T.writes(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                        for k_0_1 in T.serial(4):
                            for ax0_0, ax1_0 in T.grid(4, 1):
                                with T.block("A_global_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(1024, i_0_0 * 16 + i_0_1 * 4 + ax0_0)
                                    v1_o = T.axis.spatial(1024, k_0_0 * 4 + k_0_1 + ax1_0)
                                    T.reads(A_global_shared[v0_o, v1_o, 0 : 16, 0 : 16])
                                    T.writes(A_global_shared_wmma_matrix_a[v0_o, v1_o, 0 : 16, 0 : 16])
                                    A_1 = T.match_buffer(A_global_shared[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="int8", strides=[A_s0, A_s1], scope="shared", offset_factor=16)
                                    C_2 = T.match_buffer(A_global_shared_wmma_matrix_a[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="int8", strides=[C_s0_1, C_s1_1], scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_2.data, 16, 16, 16, C_2.elem_offset // C_s0_1 // 16 * (C_s0_1 // 16) + C_2.elem_offset % C_s0_1 // 16, T.tvm_access_ptr(T.type_annotation(dtype="int8"), A_1.data, A_1.elem_offset, A_s0 * 16, 1, dtype="handle"), A_s0, "row_major", dtype="handle")
                            for ax0_0, ax1_0 in T.grid(4, 1):
                                with T.block("B_global_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(1024, j_0_0 * 4 + ax0_0)
                                    v1_o = T.axis.spatial(1024, k_0_0 * 4 + k_0_1 + ax1_0)
                                    T.reads(B_global_shared[v0_o, v1_o, 0 : 16, 0 : 16])
                                    T.writes(B_global_shared_wmma_matrix_b[v0_o, v1_o, 0 : 16, 0 : 16])
                                    A_2 = T.match_buffer(B_global_shared[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="int8", strides=[A_s0_1, A_s1_1], scope="shared", offset_factor=16)
                                    C_3 = T.match_buffer(B_global_shared_wmma_matrix_b[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="int8", strides=[C_s0_2, C_s1_2], scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_3.data, 16, 16, 16, C_3.elem_offset // C_s0_2 // 16 * (C_s0_2 // 16) + C_3.elem_offset % C_s0_2 // 16, T.tvm_access_ptr(T.type_annotation(dtype="int8"), A_2.data, A_2.elem_offset, A_s0_1 * 16, 1, dtype="handle"), A_s0_1, "col_major", dtype="handle")
                            for i_0_2, j_0_2 in T.grid(4, 4):
                                with T.block("B_update_o"):
                                    vi_o = T.axis.spatial(1024, i_0_0 * 16 + i_0_1 * 4 + i_0_2)
                                    vj_o = T.axis.spatial(1024, j_0_0 * 4 + j_0_1 * 4 + j_0_2)
                                    vk_o = T.axis.reduce(1024, k_0_0 * 4 + k_0_1)
                                    T.reads(QC_wmma_accumulator[vi_o, vj_o, 0 : 16, 0 : 16], A_global_shared_wmma_matrix_a[vi_o, vk_o, 0 : 16, 0 : 16], B_global_shared_wmma_matrix_b[vj_o, vk_o, 0 : 16, 0 : 16])
                                    T.writes(QC_wmma_accumulator[vi_o, vj_o, 0 : 16, 0 : 16])
                                    A_3 = T.match_buffer(A_global_shared_wmma_matrix_a[vi_o, vk_o, 0 : 16, 0 : 16], [16, 16], dtype="int8", strides=[A_s0_2, A_s1_2], scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_global_shared_wmma_matrix_b[vj_o, vk_o, 0 : 16, 0 : 16], [16, 16], dtype="int8", strides=[B_s0, B_s1], scope="wmma.matrix_b", offset_factor=16)
                                    C_4 = T.match_buffer(QC_wmma_accumulator[vi_o, vj_o, 0 : 16, 0 : 16], [16, 16], dtype="int32", strides=[C_s0_3, C_s1_3], scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_4.data, C_4.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16) + C_4.elem_offset % C_s0_3 // 16, A_3.data, A_3.elem_offset // A_s0_2 // 16 * (A_s0_2 // 16) + A_3.elem_offset % A_s0_2 // 16, B_1.data, B_1.elem_offset // B_s0 // 16 * (B_s0 // 16) + B_1.elem_offset % B_s0 // 16, C_4.data, C_4.elem_offset // C_s0_3 // 16 * (C_s0_3 // 16) + C_4.elem_offset % C_s0_3 // 16, dtype="handle")
                            for ax0_0 in T.thread_binding(64, thread="blockIdx.y"):
                                for ax0_1, ax1 in T.grid(256, 16384):
                                    with T.block("Pre_compute_A"):
                                        vi = T.axis.spatial(16384, ax0_0 * 256 + ax0_1)
                                        vk = T.axis.reduce(16384, ax1)
                                        T.reads(A_global_shared[vi // 16, vk // 16, vi % 16, vk % 16])
                                        T.writes(PA[vi])
                                        with T.init():
                                            PA[vi] = 0
                                        PA[vi] = PA[vi] + 4 * T.Cast("int32", A_global_shared[vi // 16, vk // 16, vi % 16, vk % 16])
                    for ax0_0, ax1_0 in T.grid(4, 4):
                        with T.block("QC_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(1024, i_0_0 * 16 + i_0_1 * 4 + ax0_0)
                            v1_o = T.axis.spatial(1024, j_0_0 * 4 + ax1_0)
                            T.reads(QC_wmma_accumulator[v0_o, v1_o, 0 : 16, 0 : 16])
                            T.writes(QC[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            A_4 = T.match_buffer(QC_wmma_accumulator[v0_o, v1_o, 0 : 16, 0 : 16], [16, 16], dtype="int32", strides=[A_s0_3, A_s1_3], scope="wmma.accumulator", offset_factor=16)
                            C_5 = T.match_buffer(QC[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="int32", strides=[C_s0_4, C_s1_4], offset_factor=16)
                            T.tvm_store_matrix_sync(A_4.data, 16, 16, 16, A_4.elem_offset // A_s0_3 // 16 * (A_s0_3 // 16) + A_4.elem_offset % A_s0_3 // 16, T.tvm_access_ptr(T.type_annotation(dtype="int32"), C_5.data, C_5.elem_offset, C_s0_4 * 16, 2, dtype="handle"), C_s0_4, "row_major", dtype="handle")
