# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(8192, 8192), "float16"], B: T.Buffer[(8192, 8192), "float16"], C: T.Buffer[(8192, 8192), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_global = T.alloc_buffer([512, 512, 16, 16], dtype="float16")
    A_global_shared = T.alloc_buffer([512, 512, 16, 16], dtype="float16", scope="shared")
    A_global_shared_warp = T.alloc_buffer([512, 512, 16, 16], dtype="float16", scope="warp")
    B_global = T.alloc_buffer([512, 512, 16, 16], dtype="float16")
    B_global_shared = T.alloc_buffer([512, 512, 16, 16], dtype="float16", scope="shared")
    B_global_shared_warp = T.alloc_buffer([512, 512, 16, 16], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([512, 512, 16, 16], dtype="float16", scope="warp")
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("B_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B[v0, v1]
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A[v0, v1]
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("A_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("A_global_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(A_global_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("B_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("B_global_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(B_global_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for i_0_0, j_0_0_0, j_0_0_1, i_0_1, j_0_1, k_0_0, k_0_1, i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(64, 2, 16, 1, 4, 256, 2, 8, 4, 16, 16, 16):
        with T.block("B"):
            vi = T.axis.spatial(8192, i_0_0 * 128 + i_0_1 * 128 + i_0_2 * 16 + i_1)
            vj = T.axis.spatial(8192, j_0_0_0 * 4096 + j_0_0_1 * 256 + j_0_1 * 64 + j_0_2 * 16 + j_1)
            vk = T.axis.reduce(8192, k_0_0 * 32 + k_0_1 * 16 + k_1)
            T.reads(A_global_shared_warp[vi // 16, vk // 16, vi % 16, vk % 16], B_global_shared_warp[vk // 16, vj // 16, vk % 16, vj % 16])
            T.writes(C_warp[vi // 16, vj // 16, vi % 16, vj % 16])
            with T.init():
                C_warp[vi // 16, vj // 16, vi % 16, vj % 16] = T.float16(0)
            C_warp[vi // 16, vj // 16, vi % 16, vj % 16] = C_warp[vi // 16, vj // 16, vi % 16, vj % 16] + A_global_shared_warp[vi // 16, vk // 16, vi % 16, vk % 16] * B_global_shared_warp[vk // 16, vj // 16, vk % 16, vj % 16]
    for ax0, ax1 in T.grid(8192, 8192):
        with T.block("C_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(C[v0, v1])
            C[v0, v1] = C_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
