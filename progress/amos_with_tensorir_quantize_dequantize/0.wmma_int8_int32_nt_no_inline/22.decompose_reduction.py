# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int8"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    QA = T.alloc_buffer([16384, 16384], dtype="int8")
    QB = T.alloc_buffer([16384, 16384], dtype="int8")
    QC = T.alloc_buffer([16384, 16384], dtype="int32")
    A_global = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8")
    QA_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    QA_shared_wmma_matrix_a = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_a")
    B_global = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8")
    QB_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    QB_shared_wmma_matrix_b = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_b")
    QC_wmma_accumulator = T.alloc_buffer([1024, 1024, 16, 16], dtype="int32", scope="wmma.accumulator")
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
        for j_0_0_0 in T.thread_binding(8, thread="blockIdx.z"):
            for j_0_0_1 in T.thread_binding(32, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(4, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                        for ax0, ax1 in T.grid(256, 16384):
                            with T.block("Quantize_A"):
                                vi = T.axis.spatial(16384, i_0_0 * 256 + ax0)
                                vj = T.axis.spatial(16384, ax1)
                                T.reads(A_global[vi // 16, vj // 16, vi % 16, vj % 16])
                                T.writes(QA[vi, vj])
                                QA[vi, vj] = T.Cast("int8", T.round(T.Cast("float32", A_global[vi // 16, vj // 16, vi % 16, vj % 16]) * T.float32(0.5), dtype="float32") - T.float32(0))
                        for ax0, ax1 in T.grid(64, 16384):
                            with T.block("Quantize_B"):
                                vi = T.axis.spatial(16384, j_0_0_0 * 2048 + j_0_0_1 * 64 + ax0)
                                vj = T.axis.spatial(16384, ax1)
                                T.reads(B_global[vi // 16, vj // 16, vi % 16, vj % 16])
                                T.writes(QB[vi, vj])
                                QB[vi, vj] = T.Cast("int8", T.round(T.Cast("float32", B_global[vi // 16, vj // 16, vi % 16, vj % 16]) * T.float32(0.10000000000000001), dtype="float32") - T.float32(0))
                        for i_0_2_init, j_0_2_init, i_1_init, j_1_init in T.grid(4, 4, 16, 16):
                            with T.block("B_init"):
                                vi = T.axis.spatial(16384, i_0_0 * 256 + i_0_1 * 64 + i_0_2_init * 16 + i_1_init)
                                vj = T.axis.spatial(16384, j_0_0_0 * 2048 + j_0_0_1 * 64 + j_0_1 * 64 + j_0_2_init * 16 + j_1_init)
                                T.reads()
                                T.writes(QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16])
                                QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = 0
                        for k_0_0 in T.serial(256):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(8):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(16):
                                                with T.block("QA_shared"):
                                                    v0 = T.axis.spatial(16384, i_0_0 * 256 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 1024 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(16384, k_0_0 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 1024 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(QA[v0, v1])
                                                    T.writes(QA_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    QA_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = QA[v0, v1]
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(2):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(16):
                                                with T.block("QB_shared"):
                                                    v0 = T.axis.spatial(16384, j_0_0_0 * 2048 + j_0_0_1 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 1024 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(16384, k_0_0 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 1024 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 16 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(QB[v0, v1])
                                                    T.writes(QB_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    QB_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = QB[v0, v1]
                            for k_0_1 in T.serial(4):
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 1, 16, 16):
                                    with T.block("QA_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(16384, i_0_0 * 256 + i_0_1 * 64 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(16384, k_0_0 * 64 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(QA_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        T.writes(QA_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        QA_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = QA_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 1, 16, 16):
                                    with T.block("QB_shared_wmma.matrix_b"):
                                        v0 = T.axis.spatial(16384, j_0_0_0 * 2048 + j_0_0_1 * 64 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(16384, k_0_0 * 64 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(QB_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        T.writes(QB_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        QB_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = QB_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                                for i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(4, 4, 16, 16, 16):
                                    with T.block("B_update"):
                                        vi = T.axis.spatial(16384, i_0_0 * 256 + i_0_1 * 64 + i_0_2 * 16 + i_1)
                                        vj = T.axis.spatial(16384, j_0_0_0 * 2048 + j_0_0_1 * 64 + j_0_1 * 64 + j_0_2 * 16 + j_1)
                                        vk = T.axis.reduce(16384, k_0_0 * 64 + k_0_1 * 16 + k_1)
                                        T.reads(QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16], QA_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16], QB_shared_wmma_matrix_b[vj // 16, vk // 16, vj % 16, vk % 16])
                                        T.writes(QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16])
                                        QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] + T.Cast("int32", QA_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16]) * T.Cast("int32", QB_shared_wmma_matrix_b[vj // 16, vk // 16, vj % 16, vk % 16])
                        for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 4, 16, 16):
                            with T.block("QC_wmma.accumulator"):
                                v0 = T.axis.spatial(16384, i_0_0 * 256 + i_0_1 * 64 + ax0_0 * 16 + ax0_1)
                                v1 = T.axis.spatial(16384, j_0_0_0 * 2048 + j_0_0_1 * 64 + ax1_0 * 16 + ax1_1)
                                T.reads(QC_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                T.writes(QC[v0, v1])
                                QC[v0, v1] = QC_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                        for ax0, ax1 in T.grid(64, 64):
                            with T.block("DeQuantize_C"):
                                vi = T.axis.spatial(16384, i_0_0 * 256 + i_0_1 * 64 + ax0)
                                vj = T.axis.spatial(16384, j_0_0_0 * 2048 + j_0_0_1 * 64 + ax1)
                                T.reads(QC[vi, vj])
                                T.writes(C[vi, vj])
                                C[vi, vj] = T.Cast("int8", T.Cast("float32", QC[vi, vj]) / T.float32(0.01) + T.float32(0))
