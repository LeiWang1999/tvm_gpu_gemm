# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1024, 512, 16, 32), "int8"], B: T.Buffer[(1024, 512, 16, 32), "int8"], C: T.Buffer[(1024, 1024, 16, 16), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([1024, 512, 16, 32], dtype="int8", scope="shared")
    A_shared_warp = T.alloc_buffer([1024, 512, 16, 32], dtype="int8", scope="warp")
    B_shared = T.alloc_buffer([1024, 512, 16, 32], dtype="int8", scope="shared")
    B_shared_warp = T.alloc_buffer([1024, 512, 16, 32], dtype="int8", scope="warp")
    C_warp = T.alloc_buffer([1024, 1024, 16, 16], dtype="int32", scope="warp")
    for ax0, ax1, ax2, ax3 in T.grid(1024, 512, 16, 32):
        with T.block("B_shared"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v0, v1, v2, v3])
            T.writes(B_shared[v0, v1, v2, v3])
            B_shared[v0, v1, v2, v3] = B[v0, v1, v2, v3]
    for ax0, ax1, ax2, ax3 in T.grid(1024, 512, 16, 32):
        with T.block("A_shared"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v0, v1, v2, v3])
            T.writes(A_shared[v0, v1, v2, v3])
            A_shared[v0, v1, v2, v3] = A[v0, v1, v2, v3]
    for ax0, ax1, ax2, ax3 in T.grid(1024, 512, 16, 32):
        with T.block("A_shared_warp"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A_shared[v0, v1, v2, v3])
            T.writes(A_shared_warp[v0, v1, v2, v3])
            A_shared_warp[v0, v1, v2, v3] = A_shared[v0, v1, v2, v3]
    for ax0, ax1, ax2, ax3 in T.grid(1024, 512, 16, 32):
        with T.block("B_shared_warp"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B_shared[v0, v1, v2, v3])
            T.writes(B_shared_warp[v0, v1, v2, v3])
            B_shared_warp[v0, v1, v2, v3] = B_shared[v0, v1, v2, v3]
    for ii_0, jj_0, ii_1, jj_1, kk_0, kk_1, ii_2, jj_2, i, j, k in T.grid(64, 256, 4, 1, 256, 2, 4, 4, 16, 16, 32):
        with T.block("B"):
            vii = T.axis.spatial(1024, ii_0 * 16 + ii_1 * 4 + ii_2)
            vjj = T.axis.spatial(1024, jj_0 * 4 + jj_1 * 4 + jj_2)
            vkk = T.axis.reduce(512, kk_0 * 2 + kk_1)
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A_shared_warp[vii, vkk, vi, vk], B_shared_warp[vjj, vkk, vj, vk])
            T.writes(C_warp[vii, vjj, vi, vj])
            with T.init():
                C_warp[vii, vjj, vi, vj] = 0
            C_warp[vii, vjj, vi, vj] = C_warp[vii, vjj, vi, vj] + T.Cast("int32", A_shared_warp[vii, vkk, vi, vk]) * T.Cast("int32", B_shared_warp[vjj, vkk, vj, vk])
    for ax0, ax1, ax2, ax3 in T.grid(1024, 1024, 16, 16):
        with T.block("C_warp"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(C_warp[v0, v1, v2, v3])
            T.writes(C[v0, v1, v2, v3])
            C[v0, v1, v2, v3] = C_warp[v0, v1, v2, v3]
