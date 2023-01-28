# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    PA = T.alloc_buffer([16384], dtype="int32")
    PB = T.alloc_buffer([16384], dtype="int32")
    QC = T.alloc_buffer([16384, 16384], dtype="int32")
    A_global = T.alloc_buffer([16384, 16384], dtype="int8")
    A_global_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    A_global_shared_wmma_matrix_a = T.alloc_buffer([16384, 16384], dtype="int8", scope="wmma.matrix_a")
    B_global = T.alloc_buffer([16384, 16384], dtype="int8")
    B_global_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    B_global_shared_wmma_matrix_b = T.alloc_buffer([16384, 16384], dtype="int8", scope="wmma.matrix_b")
    QC_wmma_accumulator = T.alloc_buffer([16384, 16384], dtype="int32", scope="wmma.accumulator")
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
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global[v0, v1])
            T.writes(A_global_shared[v0, v1])
            A_global_shared[v0, v1] = A_global[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_global_shared_wmma.matrix_a"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global_shared[v0, v1])
            T.writes(A_global_shared_wmma_matrix_a[v0, v1])
            A_global_shared_wmma_matrix_a[v0, v1] = A_global_shared[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global[v0, v1])
            T.writes(B_global_shared[v0, v1])
            B_global_shared[v0, v1] = B_global[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_global_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global_shared[v0, v1])
            T.writes(B_global_shared_wmma_matrix_b[v0, v1])
            B_global_shared_wmma_matrix_b[v0, v1] = B_global_shared[v0, v1]
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A_global_shared_wmma_matrix_a[vi, vk], B_global_shared_wmma_matrix_b[vj, vk])
            T.writes(QC_wmma_accumulator[vi, vj])
            with T.init():
                QC_wmma_accumulator[vi, vj] = 0
            QC_wmma_accumulator[vi, vj] = QC_wmma_accumulator[vi, vj] + T.Cast("int32", A_global_shared_wmma_matrix_a[vi, vk]) * T.Cast("int32", B_global_shared_wmma_matrix_b[vj, vk])
    for i, k in T.grid(16384, 16384):
        with T.block("Pre_compute_A"):
            vi, vk = T.axis.remap("SR", [i, k])
            T.reads(A_global_shared[vi, vk])
            T.writes(PA[vi])
            with T.init():
                PA[vi] = 0
            PA[vi] = PA[vi] + 4 * T.Cast("int32", A_global_shared[vi, vk])
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("QC_wmma.accumulator"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(QC_wmma_accumulator[v0, v1])
            T.writes(QC[v0, v1])
            QC[v0, v1] = QC_wmma_accumulator[v0, v1]
