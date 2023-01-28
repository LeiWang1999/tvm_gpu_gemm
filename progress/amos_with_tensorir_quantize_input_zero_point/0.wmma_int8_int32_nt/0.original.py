# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"], PB: T.Buffer[16384, "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    QC = T.alloc_buffer([16384, 16384], dtype="int32", scope="shared")
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A[vi, vk], B[vk, vj])
            T.writes(QC[vi, vj])
            with T.init():
                QC[vi, vj] = 0
            QC[vi, vj] = QC[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast("int32", B[vk, vj])
    for i, j in T.grid(16384, 16384):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(QC[vi, vj], PB[vj])
            T.writes(C[vi, vj])
            C[vi, vj] = QC[vi, vj] + 12 + PB[vj]
