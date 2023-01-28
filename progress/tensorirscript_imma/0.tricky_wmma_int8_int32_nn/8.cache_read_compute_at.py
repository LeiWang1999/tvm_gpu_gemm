# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1024, 1024, 16, 16), "int8"], B: T.Buffer[(1024, 1024, 16, 16), "int8"], C: T.Buffer[(1024, 1024, 16, 16), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    A_shared_wmma_matrix_a = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_a")
    B_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="shared")
    B_shared_wmma_matrix_b = T.alloc_buffer([1024, 1024, 16, 16], dtype="int8", scope="wmma.matrix_b")
    C_wmma_accumulator = T.alloc_buffer([1024, 1024, 16, 16], dtype="int32", scope="wmma.accumulator")
    for ii_0 in T.thread_binding(64, thread="blockIdx.y"):
        for jj_0_0 in T.thread_binding(16, thread="blockIdx.z"):
            for jj_0_1 in T.thread_binding(16, thread="blockIdx.x"):
                for ii_1 in T.thread_binding(1, thread="threadIdx.y"):
                    for jj_1 in T.thread_binding(4, thread="threadIdx.z"):
                        for kk_0 in T.serial(512):
                            for ax0, ax1, ax2, ax3 in T.grid(16, 2, 16, 16):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(1024, ii_0 * 16 + ax0)
                                    v1 = T.axis.spatial(1024, kk_0 * 2 + ax1)
                                    v2, v3 = T.axis.remap("SS", [ax2, ax3])
                                    T.reads(A[v0, v1, v2, v3])
                                    T.writes(A_shared[v0, v1, v2, v3])
                                    A_shared[v0, v1, v2, v3] = A[v0, v1, v2, v3]
                            for ax0, ax1, ax2, ax3 in T.grid(2, 4, 16, 16):
                                with T.block("B_shared"):
                                    v0 = T.axis.spatial(1024, kk_0 * 2 + ax0)
                                    v1 = T.axis.spatial(1024, jj_0_0 * 64 + jj_0_1 * 4 + ax1)
                                    v2, v3 = T.axis.remap("SS", [ax2, ax3])
                                    T.reads(B[v0, v1, v2, v3])
                                    T.writes(B_shared[v0, v1, v2, v3])
                                    B_shared[v0, v1, v2, v3] = B[v0, v1, v2, v3]
                            for kk_1 in T.serial(2):
                                for ax0, ax1, ax2 in T.grid(16, 16, 16):
                                    with T.block("A_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(1024, ii_0 * 16 + ax0)
                                        v1 = T.axis.spatial(1024, kk_0 * 2 + kk_1)
                                        v2, v3 = T.axis.remap("SS", [ax1, ax2])
                                        T.reads(A_shared[v0, v1, v2, v3])
                                        T.writes(A_shared_wmma_matrix_a[v0, v1, v2, v3])
                                        A_shared_wmma_matrix_a[v0, v1, v2, v3] = A_shared[v0, v1, v2, v3]
                                for ax0, ax1 in T.grid(16, 16):
                                    with T.block("B_shared_wmma.matrix_b"):
                                        v0 = T.axis.spatial(1024, kk_0 * 2 + kk_1)
                                        v1 = T.axis.spatial(1024, jj_0_0 * 64 + jj_0_1 * 4 + jj_1)
                                        v2, v3 = T.axis.remap("SS", [ax0, ax1])
                                        T.reads(B_shared[v0, v1, v2, v3])
                                        T.writes(B_shared_wmma_matrix_b[v0, v1, v2, v3])
                                        B_shared_wmma_matrix_b[v0, v1, v2, v3] = B_shared[v0, v1, v2, v3]
                                for ii_2, jj_2, i, j, k in T.grid(16, 1, 16, 16, 16):
                                    with T.block("B"):
                                        vii = T.axis.spatial(1024, ii_0 * 16 + ii_1 * 16 + ii_2)
                                        vjj = T.axis.spatial(1024, jj_2 + jj_0_0 * 64 + jj_0_1 * 4 + jj_1)
                                        vkk = T.axis.reduce(1024, kk_0 * 2 + kk_1)
                                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                                        T.reads(A_shared_wmma_matrix_a[vii, vkk, vi, vk], B_shared_wmma_matrix_b[vkk, vjj, vk, vj])
                                        T.writes(C_wmma_accumulator[vii, vjj, vi, vj])
                                        with T.init():
                                            C_wmma_accumulator[vii, vjj, vi, vj] = 0
                                        C_wmma_accumulator[vii, vjj, vi, vj] = C_wmma_accumulator[vii, vjj, vi, vj] + T.Cast("int32", A_shared_wmma_matrix_a[vii, vkk, vi, vk]) * T.Cast("int32", B_shared_wmma_matrix_b[vkk, vjj, vk, vj])
                        for ax0, ax1, ax2 in T.grid(16, 16, 16):
                            with T.block("C_wmma.accumulator"):
                                v0 = T.axis.spatial(1024, ii_0 * 16 + ax0)
                                v1 = T.axis.spatial(1024, jj_0_0 * 64 + jj_0_1 * 4 + jj_1)
                                v2, v3 = T.axis.remap("SS", [ax1, ax2])
                                T.reads(C_wmma_accumulator[v0, v1, v2, v3])
                                T.writes(C[v0, v1, v2, v3])
                                C[v0, v1, v2, v3] = C_wmma_accumulator[v0, v1, v2, v3]
