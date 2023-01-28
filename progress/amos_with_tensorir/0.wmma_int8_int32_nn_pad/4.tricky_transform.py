# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(3136, 576), "int8"], B: T.Buffer[(576, 64), "int8"], C: T.Buffer[(3136, 64), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    APad_global = T.alloc_buffer([200, 40, 16, 16], dtype="int8")
    APad_global_shared = T.alloc_buffer([200, 40, 16, 16], dtype="int8", scope="shared")
    APad_global_shared_wmma_matrix_a = T.alloc_buffer([200, 40, 16, 16], dtype="int8", scope="wmma.matrix_a")
    BPad_global = T.alloc_buffer([40, 8, 16, 16], dtype="int8")
    BPad_global_shared = T.alloc_buffer([40, 8, 16, 16], dtype="int8", scope="shared")
    BPad_global_shared_wmma_matrix_b = T.alloc_buffer([40, 8, 16, 16], dtype="int8", scope="wmma.matrix_b")
    CPad_shared = T.alloc_buffer([3200, 128], dtype="int32", scope="shared")
    CPad_shared_wmma_accumulator = T.alloc_buffer([200, 8, 16, 16], dtype="int32", scope="wmma.accumulator")
    for ax0, ax1 in T.grid(3200, 640):
        with T.block("APad_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(APad_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            APad_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = T.if_then_else(v0 < 3136 and v1 < 576, A[v0, v1], T.int8(0), dtype="int8")
    for ax0, ax1 in T.grid(3200, 640):
        with T.block("APad_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(APad_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(APad_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            APad_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = APad_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(3200, 640):
        with T.block("APad_global_shared_wmma.matrix_a"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(APad_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(APad_global_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            APad_global_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = APad_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(640, 128):
        with T.block("BPad_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(BPad_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            BPad_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = T.if_then_else(v0 < 576 and v1 < 64, B[v0, v1], T.int8(0), dtype="int8")
    for ax0, ax1 in T.grid(640, 128):
        with T.block("BPad_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(BPad_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(BPad_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            BPad_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = BPad_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(640, 128):
        with T.block("BPad_global_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(BPad_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(BPad_global_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            BPad_global_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = BPad_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for i, j, k in T.grid(3200, 128, 640):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(APad_global_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16], BPad_global_shared_wmma_matrix_b[vk // 16, vj // 16, vk % 16, vj % 16])
            T.writes(CPad_shared_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16])
            with T.init():
                CPad_shared_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = 0
            CPad_shared_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = CPad_shared_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] + T.Cast("int32", APad_global_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16]) * T.Cast("int32", BPad_global_shared_wmma_matrix_b[vk // 16, vj // 16, vk % 16, vj % 16])
    for ax0, ax1 in T.grid(3200, 128):
        with T.block("CPad_shared_wmma.accumulator"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(CPad_shared_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(CPad_shared[v0, v1])
            CPad_shared[v0, v1] = CPad_shared_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(3200, 128):
        with T.block("CPad_shared"):
            T.where(ax0 < 3136 and ax1 < 64)
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(CPad_shared[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = CPad_shared[v0, v1]
