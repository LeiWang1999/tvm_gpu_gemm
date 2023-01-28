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
    for i, j in T.grid(16384, 16384):
        with T.block("Quantize_A"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(QA[vi, vj])
            QA[vi, vj] = T.Cast("int8", T.round(T.Cast("float32", A[vi, vj]) * T.float32(0.5), dtype="float32") - T.float32(0))
    for i, j in T.grid(16384, 16384):
        with T.block("Quantize_B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[vi, vj])
            T.writes(QB[vi, vj])
            QB[vi, vj] = T.Cast("int8", T.round(T.Cast("float32", B[vi, vj]) * T.float32(0.10000000000000001), dtype="float32") - T.float32(0))
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(QA[vi, vk], QB[vj, vk])
            T.writes(QC[vi, vj])
            with T.init():
                QC[vi, vj] = 0
            QC[vi, vj] = QC[vi, vj] + T.Cast("int32", QA[vi, vk]) * T.Cast("int32", QB[vj, vk])
    for i, j in T.grid(16384, 16384):
        with T.block("DeQuantize_C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(QC[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = T.Cast("int8", T.Cast("float32", QC[vi, vj]) / T.float32(0.01) + T.float32(0))
