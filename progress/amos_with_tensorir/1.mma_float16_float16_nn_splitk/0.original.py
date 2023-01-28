# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1024, 16384), "float16"], B: T.Buffer[(16384, 1024), "float16"], C: T.Buffer[(1024, 1024), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    TC = T.alloc_buffer([1, 1024, 1024], dtype="float16", scope="shared")
    for sk, i, j, k in T.grid(1, 1024, 1024, 16384):
        with T.block("B"):
            vsk, vi, vj, vk = T.axis.remap("SSSR", [sk, i, j, k])
            T.reads(A[vi, vsk * 16384 + vk], B[vsk * 16384 + vk, vj])
            T.writes(TC[vsk, vi, vj])
            with T.init():
                TC[vsk, vi, vj] = T.float16(0)
            TC[vsk, vi, vj] = TC[vsk, vi, vj] + A[vi, vsk * 16384 + vk] * B[vsk * 16384 + vk, vj]
    for sk, i, j in T.grid(1, 1024, 1024):
        with T.block("C"):
            vsk, vi, vj = T.axis.remap("SSS", [sk, i, j])
            T.reads(C[vi, vj], TC[vsk, vi, vj])
            T.writes()
            T.atomic_add(T.address_of(C[vi, vj], dtype="handle"), TC[vsk, vi, vj], dtype="float16")
