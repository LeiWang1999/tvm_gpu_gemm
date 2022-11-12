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
    for j_0, i_0, k_0, j_1, i_1, k_1 in T.grid(128, 128, 512, 128, 128, 32):
        with T.block("B"):
            vi = T.axis.spatial(16384, i_0 * 128 + i_1)
            vj = T.axis.spatial(16384, j_0 * 128 + j_1)
            vk = T.axis.reduce(16384, k_0 * 32 + k_1)
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
