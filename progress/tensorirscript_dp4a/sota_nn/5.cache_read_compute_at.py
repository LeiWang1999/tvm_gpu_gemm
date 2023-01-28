# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16384, 16384), "int8"], B: T.Buffer[(16384, 16384), "int8"], C: T.Buffer[(16384, 16384), "int32"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    A_shared_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    B_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    B_local_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    B_local_local_shared = T.alloc_buffer([16384, 16384], dtype="int8", scope="shared")
    B_local_local_shared_local = T.alloc_buffer([16384, 16384], dtype="int8", scope="local")
    C_local = T.alloc_buffer([16384, 16384], dtype="int32", scope="local")
    for i_0 in T.thread_binding(128, thread="blockIdx.y"):
        for j_0 in T.thread_binding(128, thread="blockIdx.x"):
            for k_0 in T.serial(512):
                for ax0, ax1 in T.grid(128, 32):
                    with T.block("A_shared"):
                        v0 = T.axis.spatial(16384, i_0 * 128 + ax0)
                        v1 = T.axis.spatial(16384, k_0 * 32 + ax1)
                        T.reads(A[v0, v1])
                        T.writes(A_shared[v0, v1])
                        A_shared[v0, v1] = A[v0, v1]
                for ax0, ax1 in T.grid(32, 128):
                    with T.block("B_local"):
                        v0 = T.axis.spatial(16384, k_0 * 32 + ax0)
                        v1 = T.axis.spatial(16384, j_0 * 128 + ax1)
                        T.reads(B[v0, v1])
                        T.writes(B_local[v0, v1])
                        B_local[v0, v1] = B[v0, v1]
                for ax0, ax1 in T.grid(32, 128):
                    with T.block("B_local_local"):
                        v0 = T.axis.spatial(16384, k_0 * 32 + ax0)
                        v1 = T.axis.spatial(16384, j_0 * 128 + ax1)
                        T.reads(B_local[v0, v1])
                        T.writes(B_local_local[v0, v1])
                        B_local_local[v0, v1] = B_local[v0, v1]
                for ax0, ax1 in T.grid(32, 128):
                    with T.block("B_local_local_shared"):
                        v0 = T.axis.spatial(16384, k_0 * 32 + ax0)
                        v1 = T.axis.spatial(16384, j_0 * 128 + ax1)
                        T.reads(B_local_local[v0, v1])
                        T.writes(B_local_local_shared[v0, v1])
                        B_local_local_shared[v0, v1] = B_local_local[v0, v1]
                for i_1_0 in T.thread_binding(16, thread="threadIdx.y"):
                    for j_1_0 in T.thread_binding(16, thread="threadIdx.x"):
                        for i_1_1, j_1_1, k_1_0 in T.grid(8, 8, 8):
                            for ax0 in T.serial(4):
                                with T.block("A_shared_local"):
                                    v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 8 + i_1_1)
                                    v1 = T.axis.spatial(16384, k_0 * 32 + k_1_0 * 4 + ax0)
                                    T.reads(A_shared[v0, v1])
                                    T.writes(A_shared_local[v0, v1])
                                    A_shared_local[v0, v1] = A_shared[v0, v1]
                            for ax0 in T.serial(4):
                                with T.block("B_local_local_shared_local"):
                                    v0 = T.axis.spatial(16384, k_0 * 32 + k_1_0 * 4 + ax0)
                                    v1 = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 8 + j_1_1)
                                    T.reads(B_local_local_shared[v0, v1])
                                    T.writes(B_local_local_shared_local[v0, v1])
                                    B_local_local_shared_local[v0, v1] = B_local_local_shared[v0, v1]
                            for k_1_1 in T.serial(4):
                                with T.block("B"):
                                    vi = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 8 + i_1_1)
                                    vj = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 8 + j_1_1)
                                    vk = T.axis.reduce(16384, k_0 * 32 + k_1_0 * 4 + k_1_1)
                                    T.reads(A_shared_local[vi, vk], B_local_local_shared_local[vk, vj])
                                    T.writes(C_local[vi, vj])
                                    with T.init():
                                        C_local[vi, vj] = T.float32(0)
                                    C_local[vi, vj] = C_local[vi, vj] + T.Cast("int32", A_shared_local[vi, vk]) * T.Cast("int32", B_local_local_shared_local[vk, vj])
                        for ax0, ax1 in T.grid(8, 8):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(16384, i_0 * 128 + i_1_0 * 8 + ax0)
                                v1 = T.axis.spatial(16384, j_0 * 128 + j_1_0 * 8 + ax1)
                                T.reads(C_local[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_local[v0, v1]
