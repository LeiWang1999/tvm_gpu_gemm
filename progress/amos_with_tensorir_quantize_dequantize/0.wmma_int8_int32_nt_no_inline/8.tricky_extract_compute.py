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
    for i, j in T.grid(16384, 16384):
        with T.block("Quantize_A"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A_global[vi // 16, vj // 16, vi % 16, vj % 16])
            T.writes(QA[vi, vj])
            QA[vi, vj] = T.Cast("int8", T.round(T.Cast("float32", A_global[vi // 16, vj // 16, vi % 16, vj % 16]) * T.float32(0.5), dtype="float32") - T.float32(0))
    for i, j in T.grid(16384, 16384):
        with T.block("Quantize_B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B_global[vi // 16, vj // 16, vi % 16, vj % 16])
            T.writes(QB[vi, vj])
            QB[vi, vj] = T.Cast("int8", T.round(T.Cast("float32", B_global[vi // 16, vj // 16, vi % 16, vj % 16]) * T.float32(0.10000000000000001), dtype="float32") - T.float32(0))
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QA_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QA[v0, v1])
            T.writes(QA_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            QA_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = QA[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QA_shared_wmma.matrix_a"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QA_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(QA_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            QA_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = QA_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QB_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QB[v0, v1])
            T.writes(QB_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            QB_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = QB[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QB_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QB_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(QB_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            QB_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = QB_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for i_0, j_0, k_0, i_1, j_1, k_1 in T.grid(1024, 1024, 1024, 16, 16, 16):
        with T.block("B"):
            vi = T.axis.spatial(16384, i_0 * 16 + i_1)
            vj = T.axis.spatial(16384, j_0 * 16 + j_1)
            vk = T.axis.reduce(16384, k_0 * 16 + k_1)
            T.reads(QA_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16], QB_shared_wmma_matrix_b[vj // 16, vk // 16, vj % 16, vk % 16])
            T.writes(QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16])
            with T.init():
                QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = 0
            QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = QC_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] + T.Cast("int32", QA_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16]) * T.Cast("int32", QB_shared_wmma_matrix_b[vj // 16, vk // 16, vj % 16, vk % 16])
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QC_wmma.accumulator"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QC_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(QC[v0, v1])
            QC[v0, v1] = QC_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for i, j in T.grid(16384, 16384):
        with T.block("DeQuantize_C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(QC[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = T.Cast("int8", T.Cast("float32", QC[vi, vj]) / T.float32(0.01) + T.float32(0))
