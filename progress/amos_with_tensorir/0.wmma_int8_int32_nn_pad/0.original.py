# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(3136, 576), "int8"], B: T.Buffer[(576, 64), "int8"], C: T.Buffer[(3136, 64), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    APad = T.alloc_buffer([3200, 640], dtype="int8")
    BPad = T.alloc_buffer([640, 128], dtype="int8")
    CPad = T.alloc_buffer([3200, 128], dtype="int32")
    for i, k in T.grid(3200, 640):
        with T.block("APad"):
            vi, vk = T.axis.remap("SS", [i, k])
            T.reads(A[vi, vk])
            T.writes(APad[vi, vk])
            APad[vi, vk] = T.if_then_else(vi < 3136 and vk < 576, A[vi, vk], T.int8(0), dtype="int8")
    for k, j in T.grid(640, 128):
        with T.block("BPad"):
            vk, vj = T.axis.remap("SS", [k, j])
            T.reads(B[vk, vj])
            T.writes(BPad[vk, vj])
            BPad[vk, vj] = T.if_then_else(vk < 576 and vj < 64, B[vk, vj], T.int8(0), dtype="int8")
    for i, j, k in T.grid(3200, 128, 640):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(APad[vi, vk], BPad[vk, vj])
            T.writes(CPad[vi, vj])
            with T.init():
                CPad[vi, vj] = 0
            CPad[vi, vj] = CPad[vi, vj] + T.Cast("int32", APad[vi, vk]) * T.Cast("int32", BPad[vk, vj])
    for i, j in T.grid(3136, 64):
        with T.block("CPad"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(CPad[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = CPad[vi, vj]
