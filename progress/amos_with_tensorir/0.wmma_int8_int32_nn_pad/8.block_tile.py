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
    for i_0_0, j_0_0, i_0_1, j_0_1, k_0_0, k_0_1, i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(25, 1, 1, 4, 20, 2, 8, 2, 16, 16, 16):
        with T.block("B"):
            vi = T.axis.spatial(3200, i_0_0 * 128 + i_0_1 * 128 + i_0_2 * 16 + i_1)
            vj = T.axis.spatial(128, j_0_0 * 128 + j_0_1 * 32 + j_0_2 * 16 + j_1)
            vk = T.axis.reduce(640, k_0_0 * 32 + k_0_1 * 16 + k_1)
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
