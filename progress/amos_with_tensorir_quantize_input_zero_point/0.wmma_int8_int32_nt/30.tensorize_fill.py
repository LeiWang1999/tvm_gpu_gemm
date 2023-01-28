# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"], PB: T.Buffer[16384, "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    C_s0 = T.var("int32")
    C_s1 = T.var("int32")
    # body
    # with T.block("root")
    QC = T.alloc_buffer([16384, 16384], dtype="int32", scope="shared")
    A_global = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8")
    A_global_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    A_global_shared_wmma_matrix_a = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_a")
    B_global = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8")
    B_global_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    B_global_shared_wmma_matrix_b = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_b")
    QC_wmma_accumulator = T.alloc_buffer([1024, 1024, 16, 16], dtype="int32", scope="wmma.accumulator")
    C_local = T.alloc_buffer([16384, 16384], dtype="int32", scope="local")
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A[v0, v1]
    for i_0_0 in T.thread_binding(64, thread="blockIdx.y"):
        for j_0_0_0 in T.thread_binding(16, thread="blockIdx.z"):
            for j_0_0_1 in T.thread_binding(16, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(1, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(4, thread="threadIdx.z"):
                        for i_0_2_init, j_0_2_init in T.grid(16, 1):
                            with T.block("B_init_o"):
                                vi_o = T.axis.spatial(1024, i_0_0 * 16 + i_0_1 * 16 + i_0_2_init)
                                vj_o = T.axis.spatial(1024, j_0_0_0 * 64 + j_0_0_1 * 4 + j_0_1 + j_0_2_init)
                                T.reads()
                                T.writes(QC_wmma_accumulator[vi_o, vj_o, 0 : 16, 0 : 16])
                                C_1 = T.match_buffer(QC_wmma_accumulator[vi_o, vj_o, 0 : 16, 0 : 16], [16, 16], dtype="int32", strides=[C_s0, C_s1], scope="wmma.accumulator", offset_factor=16)
                                T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_s0 // 16 * (C_s0 // 16) + C_1.elem_offset % C_s0 // 16, T.float32(0), dtype="handle")
                        for k_0_0 in T.serial(512):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(4):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(16):
                                                with T.block("A_global_shared"):
                                                    v0 = T.axis.spatial(16384, i_0_0 * 256 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 512 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(16384, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 512 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    T.writes(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(1):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(16):
                                                with T.block("B_global_shared"):
                                                    v0 = T.axis.spatial(16384, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 1024 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 1024 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    T.writes(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                            for k_0_1 in T.serial(2):
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(16, 1, 16, 16):
                                    with T.block("A_global_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(16384, i_0_0 * 256 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(16384, k_0_0 * 32 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        T.writes(A_global_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        A_global_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(1, 1, 16, 16):
                                    with T.block("B_global_shared_wmma.matrix_b"):
                                        v0 = T.axis.spatial(16384, k_0_0 * 32 + k_0_1 * 16 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 64 + j_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        T.writes(B_global_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        B_global_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                                for i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(16, 1, 16, 16, 16):
                                    with T.block("B_update"):
                                        vi = T.axis.spatial(16384, i_0_0 * 256 + i_0_1 * 256 + i_0_2 * 16 + i_1)
                                        vj = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 64 + j_0_1 * 16 + j_0_2 * 16 + j_1)
                                        vk = T.axis.reduce(16384, k_0_0 * 32 + k_0_1 * 16 + k_1)
                                        T.reads(QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16], A_global_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16], B_global_shared_wmma_matrix_b[vk // 16, vj // 16, vk % 16, vj % 16])
                                        T.writes(QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16])
                                        QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] + T.Cast("int32", A_global_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16]) * T.Cast("int32", B_global_shared_wmma_matrix_b[vk // 16, vj // 16, vk % 16, vj % 16])
                        for ax0_0, ax1_0 in T.grid(16, 1):
                            for ax0_1, ax1_1 in T.grid(16, 16):
                                with T.block("QC_wmma.accumulator"):
                                    v0 = T.axis.spatial(16384, i_0_0 * 256 + ax0_0 * 16 + ax0_1)
                                    v1 = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 64 + j_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                    T.reads(QC_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                    T.writes(QC[v0, v1])
                                    QC[v0, v1] = QC_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                            for ax0_ax1_fused_0 in T.serial(8):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        with T.block("C"):
                                            vi = T.axis.spatial(16384, i_0_0 * 256 + ax0_0 * 16 + (ax0_ax1_fused_0 * 32 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) // 16)
                                            vj = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 64 + j_0_1 * 16 + (ax0_ax1_fused_0 * 32 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) % 16)
                                            T.reads(QC[vi, vj], PB[vj])
                                            T.writes(C_local[vi, vj])
                                            C_local[vi, vj] = QC[vi, vj] + 12 + PB[vj]
                            for ax0_ax1_fused_0 in T.serial(8):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        with T.block("C_local"):
                                            v0 = T.axis.spatial(16384, i_0_0 * 256 + ax0_0 * 16 + (ax0_ax1_fused_0 * 32 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) // 16)
                                            v1 = T.axis.spatial(16384, j_0_0_0 * 1024 + j_0_0_1 * 64 + j_0_1 * 16 + (ax0_ax1_fused_0 * 32 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) % 16)
                                            T.reads(C_local[v0, v1])
                                            T.writes(C[v0, v1])
                                            C[v0, v1] = C_local[v0, v1]
