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
    for ii_0 in T.thread_binding(64, thread="blockIdx.x"):
        for jj_0 in T.thread_binding(256, thread="blockIdx.y"):
            for ii_1 in T.thread_binding(4, thread="threadIdx.y"):
                for jj_1 in T.thread_binding(1, thread="threadIdx.z"):
                    for kk_0 in T.serial(256, annotations={"thread_rasterization":16}):
                        for ax0, ax1, ax2, ax3 in T.grid(16, 2, 16, 32):
                            with T.block("A_shared"):
                                v0 = T.axis.spatial(1024, ii_0 * 16 + ax0)
                                v1 = T.axis.spatial(512, kk_0 * 2 + ax1)
                                v2, v3 = T.axis.remap("SS", [ax2, ax3])
                                T.reads(A[v0, v1, v2 % 8 * 2 + v3 // 16, v2 // 8 * 16 + v3 % 16])
                                T.writes(A_shared[v0, v1, v2, v3])
                                A_shared[v0, v1, v2, v3] = A[v0, v1, v2 % 8 * 2 + v3 // 16, v2 // 8 * 16 + v3 % 16]
                        for ax0, ax1, ax2, ax3 in T.grid(4, 2, 16, 32):
                            with T.block("B_shared"):
                                v0 = T.axis.spatial(1024, jj_0 * 4 + ax0)
                                v1 = T.axis.spatial(512, kk_0 * 2 + ax1)
                                v2, v3 = T.axis.remap("SS", [ax2, ax3])
                                T.reads(B[v0, v1, v2 // 8 * 8 + v2 % 4 * 2 + v3 // 16, v2 % 8 // 4 * 16 + v3 % 16])
                                T.writes(B_shared[v0, v1, v2, v3])
                                B_shared[v0, v1, v2, v3] = B[v0, v1, v2 // 8 * 8 + v2 % 4 * 2 + v3 // 16, v2 % 8 // 4 * 16 + v3 % 16]
                        for kk_1 in T.serial(2):
                            for ax0, ax1, ax2 in T.grid(4, 16, 32):
                                with T.block("A_shared_warp"):
                                    v0 = T.axis.spatial(1024, ii_0 * 16 + ii_1 * 4 + ax0)
                                    v1 = T.axis.spatial(512, kk_0 * 2 + kk_1)
                                    v2, v3 = T.axis.remap("SS", [ax1, ax2])
                                    T.reads(A_shared[v0, v1, v2, v3])
                                    T.writes(A_shared_warp[v0, v1, v2, v3])
                                    A_shared_warp[v0, v1, v2, v3] = A_shared[v0, v1, v2, v3]
                            for ax0, ax1, ax2 in T.grid(4, 16, 32):
                                with T.block("B_shared_warp"):
                                    v0 = T.axis.spatial(1024, jj_0 * 4 + ax0)
                                    v1 = T.axis.spatial(512, kk_0 * 2 + kk_1)
                                    v2, v3 = T.axis.remap("SS", [ax1, ax2])
                                    T.reads(B_shared[v0, v1, v2, v3])
                                    T.writes(B_shared_warp[v0, v1, v2, v3])
                                    B_shared_warp[v0, v1, v2, v3] = B_shared[v0, v1, v2, v3]
                            for ii_2, jj_2, i, j, k in T.grid(4, 4, 16, 16, 32):
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
                    for ax0, ax1, ax2, ax3 in T.grid(4, 4, 16, 16):
                        with T.block("C_warp"):
                            v0 = T.axis.spatial(1024, ii_0 * 16 + ii_1 * 4 + ax0)
                            v1 = T.axis.spatial(1024, jj_0 * 4 + ax1)
                            v2, v3 = T.axis.remap("SS", [ax2, ax3])
                            T.reads(C_warp[v0, v1, v2, v3])
                            T.writes(C[v0, v1, v2, v3])
                            C[v0, v1, v2, v3] = C_warp[v0, v1, v2, v3]
