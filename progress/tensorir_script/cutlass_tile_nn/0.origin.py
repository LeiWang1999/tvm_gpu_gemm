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
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A_local_shared_local[vi, vk], B_shared_local[vk, vj])
            T.writes(C_local[vi, vj])
            with T.init():
                C_local[vi, vj] = T.float32(0)
            C_local[vi, vj] = C_local[vi, vj] + A_local_shared_local[vi, vk] * B_shared_local[vk, vj]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_local[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_local[v0, v1]
