# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], BT: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int8"]) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    A_shared_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    BT_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    BT_shared_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    C_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("BT_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(BT[v0, v1])
            T.writes(BT_shared[v0, v1])
            BT_shared[v0, v1] = BT[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_shared[v0, v1])
            A_shared[v0, v1] = A[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_shared_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_shared[v0, v1])
            T.writes(A_shared_local[v0, v1])
            A_shared_local[v0, v1] = A_shared[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("BT_shared_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(BT_shared[v0, v1])
            T.writes(BT_shared_local[v0, v1])
            BT_shared_local[v0, v1] = BT_shared[v0, v1]
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A_shared_local[vi, vk], BT_shared_local[vj, vk])
            T.writes(C_local[vi, vj])
            with T.init():
                C_local[vi, vj] = T.float32(0)
            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vi, vk] * BT_shared_local[vj, vk]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_local[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_local[v0, v1]
