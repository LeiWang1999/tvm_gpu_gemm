# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((18966528, 32), "int8"), B: T.Buffer((1, 32), "int8"), C: T.Buffer((18966528, 1), "int32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    A_local = T.alloc_buffer((18966528, 32), "int8", scope="local")
    B_local = T.alloc_buffer((1, 32), "int8", scope="local")
    for ax0, ax1 in T.grid(1, 32):
        with T.block("B_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_local[v0, v1])
            B_local[v0, v1] = B[v0, v1]
    for ax0, ax1 in T.grid(18966528, 32):
        with T.block("A_local"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_local[v0, v1])
            A_local[v0, v1] = A[v0, v1]
    for i, j, k in T.grid(18966528, 1, 32):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A_local[vi, vk], B_local[vj, vk])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = 0
            C[vi, vj] = C[vi, vj] + T.Cast("int32", A_local[vi, vk]) * T.Cast("int32", B_local[vj, vk])