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
    A_global = T.alloc_buffer([16384, 16384], dtype="int8")
    QA_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    QA_local_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    QA_local_shared_wmma_matrix_a = T.alloc_buffer([16384, 16384], dtype="int8", scope="wmma.matrix_a")
    B_global = T.alloc_buffer([16384, 16384], dtype="int8")
    QB_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    QB_local_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    QB_local_shared_wmma_matrix_b = T.alloc_buffer([16384, 16384], dtype="int8", scope="wmma.matrix_b")
    QC_shared = T.alloc_buffer([16384, 16384], dtype="int32", scope="shared")
    QC_shared_wmma_accumulator = T.alloc_buffer([16384, 16384], dtype="int32", scope="wmma.accumulator")
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_global[v0, v1])
            B_global[v0, v1] = B[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0, v1])
            A_global[v0, v1] = A[v0, v1]
    for i, j in T.grid(16384, 16384):
        with T.block("Quantize_A"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A_global[vi, vj])
            T.writes(QA[vi, vj])
            QA[vi, vj] = T.Cast("int8", T.round(T.Cast("float32", A_global[vi, vj]) * T.float32(0.5), dtype="float32") - T.float32(0))
    for i, j in T.grid(16384, 16384):
        with T.block("Quantize_B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B_global[vi, vj])
            T.writes(QB[vi, vj])
            QB[vi, vj] = T.Cast("int8", T.round(T.Cast("float32", B_global[vi, vj]) * T.float32(0.10000000000000001), dtype="float32") - T.float32(0))
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QA_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QA[v0, v1])
            T.writes(QA_local[v0, v1])
            QA_local[v0, v1] = QA[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QA_local_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QA_local[v0, v1])
            T.writes(QA_local_shared[v0, v1])
            QA_local_shared[v0, v1] = QA_local[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QA_local_shared_wmma.matrix_a"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QA_local_shared[v0, v1])
            T.writes(QA_local_shared_wmma_matrix_a[v0, v1])
            QA_local_shared_wmma_matrix_a[v0, v1] = QA_local_shared[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QB_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QB[v0, v1])
            T.writes(QB_local[v0, v1])
            QB_local[v0, v1] = QB[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QB_local_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QB_local[v0, v1])
            T.writes(QB_local_shared[v0, v1])
            QB_local_shared[v0, v1] = QB_local[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QB_local_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QB_local_shared[v0, v1])
            T.writes(QB_local_shared_wmma_matrix_b[v0, v1])
            QB_local_shared_wmma_matrix_b[v0, v1] = QB_local_shared[v0, v1]
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(QA_local_shared_wmma_matrix_a[vi, vk], QB_local_shared_wmma_matrix_b[vj, vk])
            T.writes(QC_shared_wmma_accumulator[vi, vj])
            with T.init():
                QC_shared_wmma_accumulator[vi, vj] = 0
            QC_shared_wmma_accumulator[vi, vj] = QC_shared_wmma_accumulator[vi, vj] + T.Cast("int32", QA_local_shared_wmma_matrix_a[vi, vk]) * T.Cast("int32", QB_local_shared_wmma_matrix_b[vj, vk])
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QC_shared_wmma.accumulator"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QC_shared_wmma_accumulator[v0, v1])
            T.writes(QC_shared[v0, v1])
            QC_shared[v0, v1] = QC_shared_wmma_accumulator[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QC_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QC_shared[v0, v1])
            T.writes(QC[v0, v1])
            QC[v0, v1] = QC_shared[v0, v1]
    for i, j in T.grid(16384, 16384):
        with T.block("DeQuantize_C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(QC[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = T.Cast("int8", T.Cast("float32", QC[vi, vj]) / T.float32(0.01) + T.float32(0))
