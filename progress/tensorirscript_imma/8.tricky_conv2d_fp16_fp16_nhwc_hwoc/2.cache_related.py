# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16, 14, 14, 16, 16, 16), "float16"], W: T.Buffer[(3, 3, 16, 32, 16, 16), "float16"], Conv: T.Buffer[(16, 14, 14, 32, 16, 16), "float16"]):
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    Apad = T.alloc_buffer([16, 16, 16, 16, 16, 16], dtype="float16")
    Apad_shared = T.alloc_buffer([16, 16, 16, 16, 16, 16], dtype="float16", scope="shared")
    Apad_shared_wmma_matrix_a = T.alloc_buffer([16, 16, 16, 16, 16, 16], dtype="float16", scope="wmma.matrix_a")
    W_shared = T.alloc_buffer([3, 3, 16, 32, 16, 16], dtype="float16", scope="shared")
    W_shared_wmma_matrix_b = T.alloc_buffer([3, 3, 16, 32, 16, 16], dtype="float16", scope="wmma.matrix_b")
    Conv_wmma_accumulator = T.alloc_buffer([16, 14, 14, 32, 16, 16], dtype="float16", scope="wmma.accumulator")
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(3, 3, 16, 32, 16, 16):
        with T.block("W_shared"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(W[v0, v1, v2, v3, v4, v5])
            T.writes(W_shared[v0, v1, v2, v3, v4, v5])
            W_shared[v0, v1, v2, v3, v4, v5] = W[v0, v1, v2, v3, v4, v5]
    for n, h, w, i, nn, ii in T.grid(16, 16, 16, 16, 16, 16):
        with T.block("Apad"):
            v_n, v_h, v_w, v_i, v_nn, v_ii = T.axis.remap("SSSSSS", [n, h, w, i, nn, ii])
            T.reads(A[v_n, v_h - 1, v_w - 1, v_i, v_nn, v_ii])
            T.writes(Apad[v_n, v_h, v_w, v_i, v_nn, v_ii])
            Apad[v_n, v_h, v_w, v_i, v_nn, v_ii] = T.if_then_else(1 <= v_h and v_h < 15 and 1 <= v_w and v_w < 15, A[v_n, v_h - 1, v_w - 1, v_i, v_nn, v_ii], T.float16(0), dtype="float16")
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(16, 16, 16, 16, 16, 16):
        with T.block("Apad_shared"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(Apad[v0, v1, v2, v3, v4, v5])
            T.writes(Apad_shared[v0, v1, v2, v3, v4, v5])
            Apad_shared[v0, v1, v2, v3, v4, v5] = Apad[v0, v1, v2, v3, v4, v5]
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(16, 16, 16, 16, 16, 16):
        with T.block("Apad_shared_wmma.matrix_a"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(Apad_shared[v0, v1, v2, v3, v4, v5])
            T.writes(Apad_shared_wmma_matrix_a[v0, v1, v2, v3, v4, v5])
            Apad_shared_wmma_matrix_a[v0, v1, v2, v3, v4, v5] = Apad_shared[v0, v1, v2, v3, v4, v5]
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(3, 3, 16, 32, 16, 16):
        with T.block("W_shared_wmma.matrix_b"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(W_shared[v0, v1, v2, v3, v4, v5])
            T.writes(W_shared_wmma_matrix_b[v0, v1, v2, v3, v4, v5])
            W_shared_wmma_matrix_b[v0, v1, v2, v3, v4, v5] = W_shared[v0, v1, v2, v3, v4, v5]
    for n, h, w, o, nn, oo, ic, kh, kw, ii in T.grid(16, 14, 14, 32, 16, 16, 16, 3, 3, 16):
        with T.block("Conv"):
            v_n, v_h, v_w, v_o, v_nn, v_oo, v_ic, v_kh, v_kw, v_ii = T.axis.remap("SSSSSSRRRR", [n, h, w, o, nn, oo, ic, kh, kw, ii])
            T.reads(Apad_shared_wmma_matrix_a[v_n, v_h + v_kh, v_w + v_kw, v_ic, v_nn, v_ii], W_shared_wmma_matrix_b[v_kh, v_kw, v_ic, v_o, v_ii, v_oo])
            T.writes(Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo])
            with T.init():
                Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo] = T.float16(0)
            Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo] = Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo] + Apad_shared_wmma_matrix_a[v_n, v_h + v_kh, v_w + v_kw, v_ic, v_nn, v_ii] * W_shared_wmma_matrix_b[v_kh, v_kw, v_ic, v_o, v_ii, v_oo]
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(16, 14, 14, 32, 16, 16):
        with T.block("Conv_wmma.accumulator"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(Conv_wmma_accumulator[v0, v1, v2, v3, v4, v5])
            T.writes(Conv[v0, v1, v2, v3, v4, v5])
            Conv[v0, v1, v2, v3, v4, v5] = Conv_wmma_accumulator[v0, v1, v2, v3, v4, v5]
