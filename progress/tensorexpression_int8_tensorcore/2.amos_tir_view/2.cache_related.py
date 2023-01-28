# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_global = T.alloc_buffer([16384, 16384], dtype="int8")
    A_global_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    B_global = T.alloc_buffer([16384, 16384], dtype="int8")
    B_global_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    C_global = T.alloc_buffer([16384, 16384], dtype="int32")
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_global[v0, v1])
            B_global[v0, v1] = B[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0, v1])
            A_global[v0, v1] = A[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A_global[v0, v1])
            T.writes(A_global_shared[v0, v1])
            A_global_shared[v0, v1] = A_global[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_global[v0, v1])
            T.writes(B_global_shared[v0, v1])
            B_global_shared[v0, v1] = B_global[v0, v1]
    for i, j, k in T.grid(16384, 16384, 16384):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A_global_shared[vi, vk], B_global_shared[vj, vk])
            T.writes(C_global[vi, vj])
            with T.init():
                C_global[vi, vj] = 0
            C_global[vi, vj] = C_global[vi, vj] + T.Cast("int32", A_global_shared[vi, vk]) * T.Cast("int32", B_global_shared[vj, vk])
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_global[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_global[v0, v1]
