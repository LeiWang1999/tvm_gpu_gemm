# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "float16"], B: T.Buffer[(16384, 16384), "float16"], C: T.Buffer[(16384, 16384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="shared")
    A_shared_warp = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="warp")
    B_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="shared")
    B_shared_warp = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([1024, 1024, 16, 16], dtype="float16", scope="warp")
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(A_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(B_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for i_0, j_0, k_0, i_1, j_1, k_1 in T.grid(1024, 1024, 1024, 16, 16, 16):
        with T.block("B"):
            vi = T.axis.spatial(16384, i_0 * 16 + i_1)
            vj = T.axis.spatial(16384, j_0 * 16 + j_1)
            vk = T.axis.reduce(16384, k_0 * 16 + k_1)
            T.reads(A_shared_warp[vi // 16, vk // 16, vi % 16, vk % 16], B_shared_warp[vk // 16, vj // 16, vk % 16, vj % 16])
            T.writes(C_warp[vi // 16, vj // 16, vi % 16, vj % 16])
            with T.init():
                C_warp[vi // 16, vj // 16, vi % 16, vj % 16] = T.float16(0)
            C_warp[vi // 16, vj // 16, vi % 16, vj % 16] = C_warp[vi // 16, vj // 16, vi % 16, vj % 16] + A_shared_warp[vi // 16, vk // 16, vi % 16, vk % 16] * B_shared_warp[vk // 16, vj // 16, vk % 16, vj % 16]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(C[v0, v1])
            C[v0, v1] = C_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
