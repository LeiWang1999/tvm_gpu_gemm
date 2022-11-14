import tvm
from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(256, 256), "int8"], B: T.Buffer[(256, 256), "int8"], C: T.Buffer[(256, 256), "int32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        A_shared = T.alloc_buffer([256, 256], dtype="int8", scope="shared")
        B_local = T.alloc_buffer([256, 256], dtype="int8", scope="local")
        B_local_local = T.alloc_buffer([256, 256], dtype="int8", scope="local")
        B_local_local_shared = T.alloc_buffer(
            [256, 256], dtype="int8", scope="shared")
        C_local = T.alloc_buffer([256, 256], dtype="int32", scope="local")
        for i_0 in T.thread_binding(2, thread="blockIdx.x"):
            for j_0 in T.thread_binding(2, thread="blockIdx.y"):
                for i_1_0_init in T.thread_binding(16, thread="threadIdx.x"):
                    for j_1_0_init in T.thread_binding(16, thread="threadIdx.y"):
                        for i_1_1_init, j_1_1_init in T.grid(8, 8):
                            with T.block("B_init"):
                                vi = T.axis.spatial(
                                    256, i_0 * 128 + i_1_0_init * 8 + i_1_1_init)
                                vj = T.axis.spatial(
                                    256, j_0 * 128 + j_1_0_init * 8 + j_1_1_init)
                                T.reads()
                                T.writes(C_local[vi, vj])
                                C_local[vi, vj] = T.float32(0)
                for k_0 in T.serial(8):
                    for ax0_ax1_fused_0 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.vectorized(16):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(
                                        256, i_0 * 128 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) // 32)
                                    v1 = T.axis.spatial(
                                        256, k_0 * 32 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) % 32)
                                    T.reads(A[v0, v1])
                                    T.writes(A_shared[v0, v1])
                                    A_shared[v0, v1] = A[v0, v1]
                    for ax0_0_ax1_0_fused_0 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax0_0_ax1_0_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax0_1 in T.serial(4):
                                for ax1_1 in T.vectorized(4):
                                    with T.block("B_local"):
                                        v0 = T.axis.spatial(
                                            256, k_0 * 32 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) // 32 * 4 + ax0_1)
                                        v1 = T.axis.spatial(
                                            256, j_0 * 128 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) % 32 * 4 + ax1_1)
                                        T.reads(B[v0, v1])
                                        T.writes(B_local[v0, v1])
                                        B_local[v0, v1] = B[v0, v1]
                    for ax0_0_ax1_0_fused_0 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax0_0_ax1_0_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax0_1, ax1_1 in T.grid(4, 4):
                                with T.block("B_local_local"):
                                    v0 = T.axis.spatial(
                                        256, k_0 * 32 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) // 32 * 4 + ax0_1)
                                    v1 = T.axis.spatial(
                                        256, j_0 * 128 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) % 32 * 4 + ax1_1)
                                    T.reads(B_local[v0, v1])
                                    T.writes(
                                        B_local_local[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4])
                                    B_local_local[v0 // 4 * 4 + v1 %
                                                  4, v1 // 4 * 4 + v0 % 4] = B_local[v0, v1]
                    for ax0_0_ax1_0_fused_0 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax0_0_ax1_0_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax0_1, ax1_1 in T.grid(4, 4):
                                with T.block("B_local_local_shared"):
                                    v0 = T.axis.spatial(
                                        256, k_0 * 32 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) // 32 * 4 + ax0_1)
                                    v1 = T.axis.spatial(
                                        256, j_0 * 128 + (ax0_0_ax1_0_fused_0 * 16 + ax0_0_ax1_0_fused_1) % 32 * 4 + ax1_1)
                                    T.reads(
                                        B_local_local[v0 // 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4])
                                    T.writes(B_local_local_shared[v0, v1])
                                    B_local_local_shared[v0, v1] = B_local_local[v0 //
                                                                                 4 * 4 + v1 % 4, v1 // 4 * 4 + v0 % 4]
                    for i_1_0 in T.thread_binding(16, thread="threadIdx.x"):
                        for j_1_0 in T.thread_binding(16, thread="threadIdx.y"):
                            for i_1_1, j_1_1, k_1_0, k_1_1 in T.grid(8, 8, 8, 4):
                                with T.block("B_update"):
                                    vi = T.axis.spatial(
                                        256, i_0 * 128 + i_1_0 * 8 + i_1_1)
                                    vj = T.axis.spatial(
                                        256, j_0 * 128 + j_1_0 * 8 + j_1_1)
                                    vk = T.axis.reduce(
                                        256, k_0 * 32 + k_1_0 * 4 + k_1_1)
                                    T.reads(
                                        C_local[vi, vj], A_shared[vi, vk], B_local_local_shared[vk, vj])
                                    T.writes(C_local[vi, vj])
                                    C_local[vi, vj] = C_local[vi, vj] + T.Cast(
                                        "int32", A_shared[vi, vk]) * T.Cast("int32", B_local_local_shared[vk, vj])
                            for ax0, ax1 in T.grid(8, 8):
                                with T.block("C_local"):
                                    v0 = T.axis.spatial(
                                        256, i_0 * 128 + i_1_0 * 8 + ax0)
                                    v1 = T.axis.spatial(
                                        256, j_0 * 128 + j_1_0 * 8 + ax1)
                                    T.reads(C_local[v0, v1])
                                    T.writes(C[v0, v1])
                                    C[v0, v1] = C_local[v0, v1]


ir_module = Module
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

