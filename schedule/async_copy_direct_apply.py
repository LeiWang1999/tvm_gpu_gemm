import tvm
from intrin.async_copy import ASYNC_COPY_S8_X16_INTRIN, ASYNC_COPY_F16_X8_INTRIN
from tvm.tir.tensor_intrin.cuda import (
    WMMA_FILL_16x16x16_F16_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
)
from tvm.script import tir as T
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def func(A: T.Buffer[(128, 42, 42, 1024), "int8"], W: T.Buffer[(384, 1, 1, 1024), "int8"], Conv: T.Buffer[(225792, 384), "int32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # var definition
        tx_4 = T.env_thread("threadIdx.x")
        tx_3 = T.env_thread("threadIdx.x")
        tx_2 = T.env_thread("threadIdx.x")
        tx = T.env_thread("threadIdx.x")
        tx_1 = T.env_thread("threadIdx.x")
        C_s0 = T.var("int32")
        C_s1 = T.var("int32")
        shared_s0 = T.var("int32")
        shared_s0_1 = T.var("int32")
        shared_s1 = T.var("int32")
        shared_s1_1 = T.var("int32")
        # body
        # with T.block("root")
        data_im2col_shared = T.alloc_buffer([14112, 32, 16, 32], dtype="int8", scope="shared")
        data_im2col_shared_warp = T.alloc_buffer([14112, 32, 32, 16], dtype="int8", scope="warp")
        weight_flatten_shared = T.alloc_buffer([24, 32, 16, 32], dtype="int8", scope="shared")
        weight_flatten_shared_warp = T.alloc_buffer([24, 32, 32, 16], dtype="int8", scope="warp")
        Conv_warp = T.alloc_buffer([14112, 24, 32, 8], dtype="int32", scope="warp")
        for x_0_0 in T.thread_binding(882, thread="blockIdx.y"):
            for y_0_0 in T.thread_binding(3, thread="blockIdx.x"):
                for x_0_1 in T.thread_binding(4, thread="threadIdx.y"):
                    for y_0_1 in T.thread_binding(2, thread="threadIdx.z"):
                        for x_0_2_init, y_0_2_init in T.grid(4, 4):
                            with T.block("Conv_init_o"):
                                v_x_o = T.axis.spatial(14112, x_0_0 * 16 + x_0_1 * 4 + x_0_2_init)
                                v_y_o = T.axis.spatial(24, y_0_0 * 8 + y_0_1 * 4 + y_0_2_init)
                                T.reads()
                                T.writes(Conv_warp[v_x_o, v_y_o, 0 : 32, 0 : 8])
                                C_warp = T.match_buffer(Conv_warp[v_x_o, v_y_o, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=1)
                                T.launch_thread(tx, 32)
                                T.mma_fill(8, C_warp.data, C_warp.elem_offset, dtype="int32")
                        for k_0_0 in T.serial(16):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(8):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("data_im2col_shared"):
                                                    v0 = T.axis.spatial(225792, x_0_0 * 256 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 1024 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 512 // 32)
                                                    v1 = T.axis.spatial(1024, k_0_0 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 1024 // 512 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 32)
                                                    T.reads(A[(v0 // 16 * 16 + v0 % 8 * 2 + v1 % 32 // 16) // 1764, (v0 // 16 * 16 + v0 % 8 * 2 + v1 % 32 // 16) % 1764 // 42, (v0 // 16 * 16 + v0 % 8 * 2 + v1 % 32 // 16) % 42, v1 // 32 * 32 + (v0 // 42 * 10 + v0 % 42) % 16 // 8 * 16 + v1 % 16])
                                                    T.writes(data_im2col_shared[v0 // 16, v1 // 32, v0 % 16, v1 % 32])
                                                    data_im2col_shared[v0 // 16, v1 // 32, v0 % 16, v1 % 32] = A[(v0 // 16 * 16 + v0 % 8 * 2 + v1 % 32 // 16) // 1764, (v0 // 16 * 16 + v0 % 8 * 2 + v1 % 32 // 16) % 1764 // 42, (v0 // 16 * 16 + v0 % 8 * 2 + v1 % 32 // 16) % 42, v1 // 32 * 32 + (v0 // 42 * 10 + v0 % 42) % 16 // 8 * 16 + v1 % 16]
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(4):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("weight_flatten_shared"):
                                                    v0 = T.axis.spatial(384, y_0_0 * 128 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 1024 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 512 // 32)
                                                    v1 = T.axis.spatial(1024, k_0_0 * 64 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 1024 // 512 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 32)
                                                    T.reads(W[v0 // 8 * 8 + v0 % 4 * 2 + v1 % 32 // 16, 0, 0, v1 // 32 * 32 + v0 % 8 // 4 * 16 + v1 % 16])
                                                    T.writes(weight_flatten_shared[v0 // 16, v1 // 32, v0 % 16, v1 % 32])
                                                    weight_flatten_shared[v0 // 16, v1 // 32, v0 % 16, v1 % 32] = W[v0 // 8 * 8 + v0 % 4 * 2 + v1 % 32 // 16, 0, 0, v1 // 32 * 32 + v0 % 8 // 4 * 16 + v1 % 16]
                            for k_0_1 in T.serial(2):
                                for ax0_0, ax1_0 in T.grid(4, 1):
                                    with T.block("data_im2col_shared_warp_o"):
                                        v0_o = T.axis.spatial(14112, x_0_0 * 16 + x_0_1 * 4 + ax0_0)
                                        v1_o = T.axis.spatial(32, k_0_0 * 2 + k_0_1 + ax1_0)
                                        T.reads(data_im2col_shared[v0_o, v1_o, 0 : 16, 0 : 32])
                                        T.writes(data_im2col_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16])
                                        warp = T.match_buffer(data_im2col_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(data_im2col_shared[v0_o, v1_o, 0 : 16, 0 : 32], [16, 32], dtype="int8", strides=[shared_s0, shared_s1], scope="shared", offset_factor=16)
                                        T.launch_thread(tx_1, 32)
                                        T.ptx_ldmatrix(False, 4, ".b16", warp.data, warp.elem_offset + 16 * tx_1, T.tvm_access_ptr(T.type_annotation(dtype="int8"), shared.data, shared.elem_offset, shared_s0 * 16, 1, dtype="handle"), 16 * tx_1, dtype="int8")
                                for ax0_0, ax1_0 in T.grid(4, 1):
                                    with T.block("weight_flatten_shared_warp_o"):
                                        v0_o = T.axis.spatial(24, y_0_0 * 8 + y_0_1 * 4 + ax0_0)
                                        v1_o = T.axis.spatial(32, k_0_0 * 2 + k_0_1 + ax1_0)
                                        T.reads(weight_flatten_shared[v0_o, v1_o, 0 : 16, 0 : 32])
                                        T.writes(weight_flatten_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16])
                                        warp_1 = T.match_buffer(weight_flatten_shared_warp[v0_o, v1_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                        shared_1 = T.match_buffer(weight_flatten_shared[v0_o, v1_o, 0 : 16, 0 : 32], [16, 32], dtype="int8", strides=[shared_s0_1, shared_s1_1], scope="shared", offset_factor=16)
                                        T.launch_thread(tx_2, 32)
                                        T.ptx_ldmatrix(False, 4, ".b16", warp_1.data, warp_1.elem_offset + 16 * tx_2, T.tvm_access_ptr(T.type_annotation(dtype="int8"), shared_1.data, shared_1.elem_offset, shared_s0_1 * 16, 1, dtype="handle"), 16 * tx_2, dtype="int8")
                                for x_0_2, y_0_2 in T.grid(4, 4):
                                    with T.block("Conv_update_o"):
                                        v_x_o = T.axis.spatial(14112, x_0_0 * 16 + x_0_1 * 4 + x_0_2)
                                        v_y_o = T.axis.spatial(24, y_0_0 * 8 + y_0_1 * 4 + y_0_2)
                                        v_k_o = T.axis.reduce(32, k_0_0 * 2 + k_0_1)
                                        T.reads(Conv_warp[v_x_o, v_y_o, 0 : 32, 0 : 8], data_im2col_shared_warp[v_x_o, v_k_o, 0 : 32, 0 : 16], weight_flatten_shared_warp[v_y_o, v_k_o, 0 : 32, 0 : 16])
                                        T.writes(Conv_warp[v_x_o, v_y_o, 0 : 32, 0 : 8])
                                        A_1 = T.match_buffer(data_im2col_shared_warp[v_x_o, v_k_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                        B = T.match_buffer(weight_flatten_shared_warp[v_y_o, v_k_o, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                        C = T.match_buffer(Conv_warp[v_x_o, v_y_o, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=16)
                                        T.launch_thread(tx_3, 32)
                                        T.ptx_mma("m16n8k32", "row", "col", "int8", "int8", "int32", A_1.data, A_1.elem_offset + tx_3 * 16, B.data, B.elem_offset + tx_3 * 16, C.data, C.elem_offset + tx_3 * 8, False, dtype="int32")
                                        T.ptx_mma("m16n8k32", "row", "col", "int8", "int8", "int32", A_1.data, A_1.elem_offset + tx_3 * 16, B.data, B.elem_offset + tx_3 * 16 + T.FloorDiv(16, 2), C.data, C.elem_offset + tx_3 * 8 + T.FloorDiv(8, 2), False, dtype="int32")
                        for ax0_0, ax1_0 in T.grid(4, 4):
                            with T.block("Conv_warp_o"):
                                v0_o = T.axis.spatial(14112, x_0_0 * 16 + x_0_1 * 4 + ax0_0)
                                v1_o = T.axis.spatial(24, y_0_0 * 8 + y_0_1 * 4 + ax1_0)
                                T.reads(Conv_warp[v0_o, v1_o, 0 : 32, 0 : 8])
                                T.writes(Conv[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                C_warp_1 = T.match_buffer(Conv_warp[v0_o, v1_o, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=1)
                                C_1 = T.match_buffer(Conv[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="int32", strides=[C_s0, C_s1], offset_factor=1)
                                T.launch_thread(tx_4, 32)
                                T.mma_store(16, 16, T.tvm_access_ptr(T.type_annotation(dtype="int32"), C_1.data, C_1.elem_offset, C_s0 * 16, 2, dtype="handle"), C_warp_1.data, C_warp_1.elem_offset, C_s0, dtype="int32")


ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
print(cuda_mod.imported_modules[0].get_source())
