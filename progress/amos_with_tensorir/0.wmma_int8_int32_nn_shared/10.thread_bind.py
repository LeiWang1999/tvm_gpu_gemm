# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_global = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8")
    A_global_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    A_global_shared_wmma_matrix_a = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_a")
    B_global = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8")
    B_global_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    B_global_shared_wmma_matrix_b = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_b")
    C_shared = T.alloc_buffer([16384, 16384], dtype="int32", scope="shared")
    C_shared_wmma_accumulator = T.alloc_buffer([1024, 1024, 16, 16], dtype="int32", scope="wmma.accumulator")
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
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_global_shared_wmma.matrix_a"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(A_global_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global_shared_wmma_matrix_a[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_global_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(B_global_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for i_0_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0_0 in T.thread_binding(128, thread="blockIdx.x"):
            for i_0_1 in T.thread_binding(1, thread="threadIdx.y"):
                for j_0_1 in T.thread_binding(4, thread="threadIdx.z"):
                    for k_0_0 in T.serial(512, annotations={"thread_rasterization":1}):
                        for k_0_1, i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(2, 8, 2, 16, 16, 16):
                            with T.block("B"):
                                vi = T.axis.spatial(16384, i_0_0 * 128 + i_0_1 * 128 + i_0_2 * 16 + i_1)
                                vj = T.axis.spatial(16384, j_0_0 * 128 + j_0_1 * 32 + j_0_2 * 16 + j_1)
                                vk = T.axis.reduce(16384, k_0_0 * 32 + k_0_1 * 16 + k_1)
                                T.reads(A_global_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16], B_global_shared_wmma_matrix_b[vk // 16, vj // 16, vk % 16, vj % 16])
                                T.writes(C_shared_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16])
                                with T.init():
                                    C_shared_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = 0
                                C_shared_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] = C_shared_wmma_accumulator[vi // 16, vj // 16, vi % 16, vj % 16] + T.Cast("int32", A_global_shared_wmma_matrix_a[vi // 16, vk // 16, vi % 16, vk % 16]) * T.Cast("int32", B_global_shared_wmma_matrix_b[vk // 16, vj // 16, vk % 16, vj % 16])
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_shared_wmma.accumulator"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_shared_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(C_shared[v0, v1])
            C_shared[v0, v1] = C_shared_wmma_accumulator[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_shared[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_shared[v0, v1]
