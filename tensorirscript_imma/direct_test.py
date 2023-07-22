import tvm
from tvm.script import tir as T
from intrin.tricky_mma_float16_float16 import (
    TRICKY_MMA_fill_16x16_f16_INTRIN,
    TRICKY_LDMATRIX_16x16_A_INTRIN,
    TRICKY_LDMATRIX_16x16_B_INTRIN,
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
    TRICKY_MMA_f16f16f16_INTRIN,
    TRICKY_MMA_f16f16f16_TRANS_INTRIN,
    TRICKY_MMA_store_16x16_f16_global_INTRIN,
    A_global_16x16_to_shared_load_16x16_layout,
    B_global_16x16_to_shared_load_16x16_layout,
    C_shared_16x16_to_ldmatrix_32x8_layout,
    A_B_shared_16x16_to_ldmatrix_32x8_layout
)


@T.prim_func
def main(A: T.Buffer((8, 42, 42, 64, 16, 16), "float16"), W: T.Buffer((24, 3, 3, 64, 16, 16), "float16"), Conv: T.Buffer((8, 40, 40, 24, 16, 16), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    A_shared = T.alloc_buffer((8, 42, 42, 64, 16, 16),
                              "float16", scope="shared")
    A_shared_warp = T.alloc_buffer(
        (8, 42, 42, 64, 32, 8), "float16", scope="warp")
    W_shared = T.alloc_buffer((24, 3, 3, 64, 16, 16),
                              "float16", scope="shared")
    W_shared_warp = T.alloc_buffer(
        (24, 3, 3, 64, 32, 8), "float16", scope="warp")
    Conv_warp = T.alloc_buffer((8, 40, 40, 24, 32, 8), "float16", scope="warp")
    for n_h_fused_0 in T.thread_binding(320, thread="blockIdx.x"):
        for w_o_fused_0 in T.thread_binding(960, thread="blockIdx.y"):
            for n_h_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                for w_o_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                    for n_h_fused_2_init, w_o_fused_2_init in T.grid(1, 1):
                        with T.block("Conv_init_o"):
                            v_n = T.axis.spatial(
                                8, (n_h_fused_0 + n_h_fused_1 + n_h_fused_2_init) // 40)
                            v_h = T.axis.spatial(
                                40, (n_h_fused_0 + n_h_fused_1 + n_h_fused_2_init) % 40)
                            v_w = T.axis.spatial(
                                40, (w_o_fused_2_init + w_o_fused_0 + w_o_fused_1) // 24)
                            v_o = T.axis.spatial(
                                24, (w_o_fused_2_init + w_o_fused_0 + w_o_fused_1) % 24)
                            v_nn_o = T.axis.spatial(1, 0)
                            v_oo_o = T.axis.spatial(1, 0)
                            T.reads()
                            T.writes(Conv_warp[v_n, v_h, v_w, v_o, 0:32, 0:8])
                            C_warp = T.match_buffer(
                                Conv_warp[v_n, v_h, v_w, v_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                            tx = T.env_thread("threadIdx.x")
                            T.launch_thread(tx, 32)
                            T.mma_fill("float16", 8, C_warp.data,
                                       C_warp.elem_offset)
                    for ic_0 in range(64):
                        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 3, 3, 1, 16, 16):
                            with T.block("A_shared"):
                                v0 = T.axis.spatial(8, n_h_fused_0 // 40 + ax0)
                                v1 = T.axis.spatial(42, n_h_fused_0 % 40 + ax1)
                                v2 = T.axis.spatial(
                                    42, w_o_fused_0 // 24 + ax2)
                                v3 = T.axis.spatial(64, ic_0 + ax3)
                                v4, v5 = T.axis.remap("SS", [ax4, ax5])
                                T.reads(A[v0, v1, v2, v3, v4, v5])
                                T.writes(A_shared[v0, v1, v2, v3, v4, v5])
                                A_shared[v0, v1, v2, v3, v4,
                                         v5] = A[v0, v1, v2, v3, v4, v5]
                        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 3, 3, 1, 16, 16):
                            with T.block("W_shared"):
                                v0 = T.axis.spatial(24, w_o_fused_0 % 24 + ax0)
                                v1, v2 = T.axis.remap("SS", [ax1, ax2])
                                v3 = T.axis.spatial(64, ic_0 + ax3)
                                v4, v5 = T.axis.remap("SS", [ax4, ax5])
                                T.reads(W[v0, v1, v2, v3, v4, v5])
                                T.writes(W_shared[v0, v1, v2, v3, v4, v5])
                                W_shared[v0, v1, v2, v3, v4,
                                         v5] = W[v0, v1, v2, v3, v4, v5]
                        for ic_1, kh, kw in T.grid(1, 3, 3):
                            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 1):
                                with T.block("A_shared_warp_o"):
                                    v0 = T.axis.spatial(
                                        8, n_h_fused_0 // 40 + ax0)
                                    v1 = T.axis.spatial(
                                        42, kh + n_h_fused_0 % 40 + ax1)
                                    v2 = T.axis.spatial(
                                        42, w_o_fused_0 // 24 + kw + ax2)
                                    v3 = T.axis.spatial(64, ic_0 + ax3)
                                    v4_o = T.axis.spatial(1, 0)
                                    v5_o = T.axis.spatial(1, 0)
                                    T.reads(
                                        A_shared[v0, v1, v2, v3, 0:16, 0:16])
                                    T.writes(
                                        A_shared_warp[v0, v1, v2, v3, 0:32, 0:8])
                                    warp = T.match_buffer(
                                        A_shared_warp[v0, v1, v2, v3, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                    shared_s0 = T.int32()
                                    shared_s1 = T.int32()
                                    shared = T.match_buffer(A_shared[v0, v1, v2, v3, 0:16, 0:16], (16, 16), "float16", strides=(
                                        shared_s0, shared_s1), scope="shared", offset_factor=16)
                                    tx = T.env_thread("threadIdx.x")
                                    T.launch_thread(tx, 32)
                                    T.ptx_ldmatrix("float16", False, 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(
                                        T.type_annotation("float16"), shared.data, shared.elem_offset, shared_s0 * 16, 1), 8 * tx)
                            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 1):
                                with T.block("W_shared_warp_o"):
                                    v0 = T.axis.spatial(
                                        24, ax0 * 24 + w_o_fused_0 % 24)
                                    v1 = T.axis.spatial(3, ax1 * 3 + kh)
                                    v2 = T.axis.spatial(3, ax2 * 3 + kw)
                                    v3 = T.axis.spatial(64, ax3 * 64 + ic_0)
                                    v4_o = T.axis.spatial(1, 0)
                                    v5_o = T.axis.spatial(1, 0)
                                    T.reads(
                                        W_shared[v0, v1, v2, v3, 0:16, 0:16])
                                    T.writes(
                                        W_shared_warp[v0, v1, v2, v3, 0:32, 0:8])
                                    warp = T.match_buffer(
                                        W_shared_warp[v0, v1, v2, v3, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                    shared_s0 = T.int32()
                                    shared_s1 = T.int32()
                                    shared = T.match_buffer(W_shared[v0, v1, v2, v3, 0:16, 0:16], (16, 16), "float16", strides=(
                                        shared_s0, shared_s1), scope="shared", offset_factor=16)
                                    tx = T.env_thread("threadIdx.x")
                                    T.launch_thread(tx, 32)
                                    T.ptx_ldmatrix("float16", False, 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(
                                        T.type_annotation("float16"), shared.data, shared.elem_offset, shared_s0 * 16, 1), 8 * tx)
                            for n_h_fused_2, w_o_fused_2 in T.grid(1, 1):
                                with T.block("Conv_update_o"):
                                    v_n = T.axis.spatial(
                                        8, (n_h_fused_0 + n_h_fused_1 + n_h_fused_2) // 40)
                                    v_h = T.axis.spatial(
                                        40, (n_h_fused_0 + n_h_fused_1 + n_h_fused_2) % 40)
                                    v_w = T.axis.spatial(
                                        40, (w_o_fused_2 + w_o_fused_0 + w_o_fused_1) // 24)
                                    v_o = T.axis.spatial(
                                        24, (w_o_fused_2 + w_o_fused_0 + w_o_fused_1) % 24)
                                    v_nn_o = T.axis.spatial(1, 0)
                                    v_oo_o = T.axis.spatial(1, 0)
                                    v_ic = T.axis.reduce(64, ic_1 + ic_0)
                                    v_kh, v_kw = T.axis.remap("RR", [kh, kw])
                                    v_ii_o = T.axis.reduce(1, 0)
                                    T.reads(Conv_warp[v_n, v_h, v_w, v_o, 0:32, 0:8], A_shared_warp[v_n, v_h + v_kh,
                                            v_w + v_kw, v_ic, 0:32, 0:8], W_shared_warp[v_o, v_kh, v_kw, v_ic, 0:32, 0:8])
                                    T.writes(
                                        Conv_warp[v_n, v_h, v_w, v_o, 0:32, 0:8])
                                    A_1 = T.match_buffer(
                                        A_shared_warp[v_n, v_h + v_kh, v_w + v_kw, v_ic, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                    B = T.match_buffer(W_shared_warp[v_o, v_kh, v_kw, v_ic, 0:32, 0:8], (
                                        32, 8), "float16", scope="warp", offset_factor=16)
                                    C = T.match_buffer(Conv_warp[v_n, v_h, v_w, v_o, 0:32, 0:8], (
                                        32, 8), "float16", scope="warp", offset_factor=16)
                                    tx = T.env_thread("threadIdx.x")
                                    T.launch_thread(tx, 32)
                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data,
                                              A_1.elem_offset + tx * 8, B.data, B.elem_offset + tx * 8, C.data, C.elem_offset + tx * 8, False)
                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8,
                                              B.data, B.elem_offset + tx * 8 + T.FloorDiv(8, 2), C.data, C.elem_offset + tx * 8 + T.FloorDiv(8, 2), False)
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 1):
                        with T.block("Conv_warp_o"):
                            v0 = T.axis.spatial(8, ax0 * 8 + n_h_fused_0 // 40)
                            v1 = T.axis.spatial(
                                40, ax1 * 40 + n_h_fused_0 % 40)
                            v2 = T.axis.spatial(
                                40, ax2 * 40 + w_o_fused_0 // 24)
                            v3 = T.axis.spatial(
                                24, ax3 * 24 + w_o_fused_0 % 24)
                            v4_o = T.axis.spatial(1, 0)
                            v5_o = T.axis.spatial(1, 0)
                            T.reads(Conv_warp[v0, v1, v2, v3, 0:32, 0:8])
                            T.writes(Conv[v0, v1, v2, v3, 0:16, 0:16])
                            C_warp = T.match_buffer(
                                Conv_warp[v0, v1, v2, v3, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                            C_s0 = T.int32()
                            C_s1 = T.int32()
                            C = T.match_buffer(Conv[v0, v1, v2, v3, 0:16, 0:16], (16, 16), "float16", strides=(
                                C_s0, C_s1), offset_factor=1)
                            tx = T.env_thread("threadIdx.x")
                            T.launch_thread(tx, 32)
                            T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation(
                                "float16"), C.data, C.elem_offset, C_s0 * 16, 2), C_warp.data, C_warp.elem_offset, C_s0)


ir_module = main
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
# print(sch.mod)

cuda_mod = tvm.build(sch.mod, target="cuda")

print(cuda_mod.imported_modules[0].get_source())
