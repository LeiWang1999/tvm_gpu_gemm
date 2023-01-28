# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1024, 16384), "float16"], B: T.Buffer[(16384, 1024), "float16"], C: T.Buffer[(1024, 1024), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    TC = T.alloc_buffer([1, 1024, 1024], dtype="float16", scope="shared")
    A_global = T.alloc_buffer([64, 1024, 16, 16], dtype="float16")
    A_global_shared = T.alloc_buffer([64, 1024, 16, 16], dtype="float16", scope="shared")
    A_global_shared_warp = T.alloc_buffer([64, 1024, 16, 16], dtype="float16", scope="warp")
    B_global = T.alloc_buffer([1024, 64, 16, 16], dtype="float16")
    B_global_shared = T.alloc_buffer([1024, 64, 16, 16], dtype="float16", scope="shared")
    B_global_shared_warp = T.alloc_buffer([1024, 64, 16, 16], dtype="float16", scope="warp")
    TC_warp = T.alloc_buffer([1, 64, 64, 16, 16], dtype="float16", scope="warp")
    for ax0, ax1 in T.grid(16384, 1024):
        with T.block("B_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B[v0, v1]
    for ax0, ax1 in T.grid(1024, 16384):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A[v0, v1]
    for ax0, ax1 in T.grid(1024, 16384):
        with T.block("A_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(1024, 16384):
        with T.block("A_global_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(A_global_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            A_global_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(16384, 1024):
        with T.block("B_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(16384, 1024):
        with T.block("B_global_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(B_global_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_global_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for sk, i, j, k in T.grid(1, 1024, 1024, 16384):
        with T.block("B"):
            vsk, vi, vj, vk = T.axis.remap("SSSR", [sk, i, j, k])
            T.reads(A_global_shared_warp[vi // 16, vk // 16, vi % 16, vk % 16], B_global_shared_warp[vk // 16, vj // 16, vk % 16, vj % 16])
            T.writes(TC_warp[vsk, vi // 16, vj // 16, vi % 16, vj % 16])
            with T.init():
                TC_warp[vsk, vi // 16, vj // 16, vi % 16, vj % 16] = T.float16(0)
            TC_warp[vsk, vi // 16, vj // 16, vi % 16, vj % 16] = TC_warp[vsk, vi // 16, vj // 16, vi % 16, vj % 16] + A_global_shared_warp[vi // 16, vk // 16, vi % 16, vk % 16] * B_global_shared_warp[vk // 16, vj // 16, vk % 16, vj % 16]
    for ax0, ax1, ax2 in T.grid(1, 1024, 1024):
        with T.block("TC_warp"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(TC_warp[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            T.writes(TC[v0, v1, v2])
            TC[v0, v1, v2] = TC_warp[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16]
    for sk, i, j in T.grid(1, 1024, 1024):
        with T.block("C"):
            vsk, vi, vj = T.axis.remap("SSS", [sk, i, j])
            T.reads(C[vi, vj], TC[vsk, vi, vj])
            T.writes()
            T.atomic_add(T.address_of(C[vi, vj], dtype="handle"), TC[vsk, vi, vj], dtype="float16")
