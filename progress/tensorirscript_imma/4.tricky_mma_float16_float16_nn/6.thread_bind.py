# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(196, 36, 16, 16), "float16"], B: T.Buffer[(36, 4, 16, 16), "float16"], C: T.Buffer[(1, 196, 4, 16, 16), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([196, 36, 16, 16], dtype="float16", scope="shared")
    A_shared_warp = T.alloc_buffer([196, 36, 16, 16], dtype="float16", scope="warp")
    B_shared = T.alloc_buffer([36, 4, 16, 16], dtype="float16", scope="shared")
    B_shared_warp = T.alloc_buffer([36, 4, 16, 16], dtype="float16", scope="warp")
    C_warp = T.alloc_buffer([1, 196, 4, 16, 16], dtype="float16", scope="warp")
    for ax0, ax1, ax2, ax3 in T.grid(36, 4, 16, 16):
        with T.block("B_shared"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v0, v1, v2, v3])
            T.writes(B_shared[v0, v1, v2, v3])
            B_shared[v0, v1, v2, v3] = B[v0, v1, v2, v3]
    for ax0, ax1, ax2, ax3 in T.grid(196, 36, 16, 16):
        with T.block("A_shared"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v0, v1, v2, v3])
            T.writes(A_shared[v0, v1, v2, v3])
            A_shared[v0, v1, v2, v3] = A[v0, v1, v2, v3]
    for ax0, ax1, ax2, ax3 in T.grid(196, 36, 16, 16):
        with T.block("A_shared_warp"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A_shared[v0, v1, v2, v3])
            T.writes(A_shared_warp[v0, v1, v2, v3])
            A_shared_warp[v0, v1, v2, v3] = A_shared[v0, v1, v2, v3]
    for ax0, ax1, ax2, ax3 in T.grid(36, 4, 16, 16):
        with T.block("B_shared_warp"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B_shared[v0, v1, v2, v3])
            T.writes(B_shared_warp[v0, v1, v2, v3])
            B_shared_warp[v0, v1, v2, v3] = B_shared[v0, v1, v2, v3]
    for sk in T.thread_binding(1, thread="blockIdx.z"):
        for ii_0 in T.thread_binding(49, thread="blockIdx.y"):
            for jj_0 in T.thread_binding(1, thread="blockIdx.x"):
                for ii_1 in T.thread_binding(2, thread="threadIdx.y"):
                    for jj_1 in T.thread_binding(2, thread="threadIdx.z"):
                        for kk_0, kk_1, ii_2, jj_2, i, j, k in T.grid(18, 2, 2, 2, 16, 16, 16):
                            with T.block("B"):
                                vsk = T.axis.spatial(1, sk)
                                vii = T.axis.spatial(196, ii_0 * 4 + ii_1 * 2 + ii_2)
                                vjj = T.axis.spatial(4, jj_0 * 4 + jj_1 * 2 + jj_2)
                                vkk = T.axis.reduce(36, kk_0 * 2 + kk_1)
                                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                                T.reads(A_shared_warp[vii, vkk, vi, vk], B_shared_warp[vkk, vjj, vk, vj])
                                T.writes(C_warp[vsk, vii, vjj, vi, vj])
                                with T.init():
                                    C_warp[vsk, vii, vjj, vi, vj] = T.float32(0)
                                C_warp[vsk, vii, vjj, vi, vj] = C_warp[vsk, vii, vjj, vi, vj] + A_shared_warp[vii, vkk, vi, vk] * B_shared_warp[vkk, vjj, vk, vj]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 196, 4, 16, 16):
        with T.block("C_warp"):
            v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(C_warp[v0, v1, v2, v3, v4])
            T.writes(C[v0, v1, v2, v3, v4])
            C[v0, v1, v2, v3, v4] = C_warp[v0, v1, v2, v3, v4]
