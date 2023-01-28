# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "float32"], B: T.Buffer[(16384, 16384), "float32"], C: T.Buffer[(16384, 16384), "float32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    A_local_shared = T.alloc_buffer([16384, 16384], dtype="float32", scope="shared")
    A_local_shared_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    B_shared = T.alloc_buffer([16384, 16384], dtype="float32", scope="shared")
    B_shared_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    C_local = T.alloc_buffer([16384, 16384], dtype="float32", scope="local")
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_shared[v0, v1])
            B_shared[v0, v1] = B[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_local[v0, v1])
            A_local[v0, v1] = A[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_local_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_local[v0, v1])
            T.writes(A_local_shared[v0, v1])
            A_local_shared[v0, v1] = A_local[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_local_shared_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_local_shared[v0, v1])
            T.writes(A_local_shared_local[v0, v1])
            A_local_shared_local[v0, v1] = A_local_shared[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_shared_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_shared[v0, v1])
            T.writes(B_shared_local[v0, v1])
            B_shared_local[v0, v1] = B_shared[v0, v1]
    for i_0 in T.thread_binding(512, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for i_1 in T.thread_binding(2, thread="vthread.y"):
                for j_1 in T.thread_binding(2, thread="vthread.x"):
                    for i_2 in T.thread_binding(4, thread="threadIdx.y"):
                        for j_2 in T.thread_binding(32, thread="threadIdx.x"):
                            for k_0, k_1, i_3, j_3 in T.grid(512, 32, 4, 4):
                                with T.block("B"):
                                    vi = T.axis.spatial(16384, i_0 * 32 + i_1 * 16 + i_2 * 4 + i_3)
                                    vj = T.axis.spatial(16384, j_0 * 256 + j_1 * 128 + j_2 * 4 + j_3)
                                    vk = T.axis.reduce(16384, k_0 * 32 + k_1)
                                    T.reads(A_local_shared_local[vi, vk], B_shared_local[vk, vj])
                                    T.writes(C_local[vi, vj])
                                    with T.init():
                                        C_local[vi, vj] = T.float32(0)
                                    C_local[vi, vj] = C_local[vi, vj] + A_local_shared_local[vi, vk] * B_shared_local[vk, vj]
                            for ax0, ax1 in T.grid(4, 4):
                                with T.block("C_local"):
                                    v0 = T.axis.spatial(16384, i_0 * 32 + i_1 * 16 + i_2 * 4 + ax0)
                                    v1 = T.axis.spatial(16384, j_0 * 256 + j_1 * 128 + j_2 * 4 + ax1)
                                    T.reads(C_local[v0, v1])
                                    T.writes(C[v0, v1])
                                    C[v0, v1] = C_local[v0, v1]
