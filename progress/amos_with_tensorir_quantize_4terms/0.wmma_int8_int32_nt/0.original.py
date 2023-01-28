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
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A[vi, vk], B[vj, vk])
            T.writes(QC[vi, vj])
            with T.init():
                QC[vi, vj] = 0
            QC[vi, vj] = QC[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast("int32", B[vj, vk])
    for i, k in T.grid(16384, 16384):
        with T.block("Pre_compute_A"):
            vi, vk = T.axis.remap("SR", [i, k])
            T.reads(A[vi, vk])
            T.writes(PA[vi])
            with T.init():
                PA[vi] = 0
            PA[vi] = PA[vi] + 4 * T.Cast("int32", A[vi, vk])
