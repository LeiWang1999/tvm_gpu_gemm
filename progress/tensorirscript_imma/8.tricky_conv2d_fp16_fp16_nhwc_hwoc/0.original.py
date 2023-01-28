# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16, 14, 14, 16, 16, 16), "float16"], W: T.Buffer[(3, 3, 16, 32, 16, 16), "float16"], Conv: T.Buffer[(16, 14, 14, 32, 16, 16), "float16"]):
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    Apad = T.alloc_buffer([16, 16, 16, 16, 16, 16], dtype="float16")
    for n, h, w, i, nn, ii in T.grid(16, 16, 16, 16, 16, 16):
        with T.block("Apad"):
            v_n, v_h, v_w, v_i, v_nn, v_ii = T.axis.remap("SSSSSS", [n, h, w, i, nn, ii])
            T.reads(A[v_n, v_h - 1, v_w - 1, v_i, v_nn, v_ii])
            T.writes(Apad[v_n, v_h, v_w, v_i, v_nn, v_ii])
            Apad[v_n, v_h, v_w, v_i, v_nn, v_ii] = T.if_then_else(1 <= v_h and v_h < 15 and 1 <= v_w and v_w < 15, A[v_n, v_h - 1, v_w - 1, v_i, v_nn, v_ii], T.float16(0), dtype="float16")
    for n, h, w, o, nn, oo, ic, kh, kw, ii in T.grid(16, 14, 14, 32, 16, 16, 16, 3, 3, 16):
        with T.block("Conv"):
            v_n, v_h, v_w, v_o, v_nn, v_oo, v_ic, v_kh, v_kw, v_ii = T.axis.remap("SSSSSSRRRR", [n, h, w, o, nn, oo, ic, kh, kw, ii])
            T.reads(Apad[v_n, v_h + v_kh, v_w + v_kw, v_ic, v_nn, v_ii], W[v_kh, v_kw, v_ic, v_o, v_ii, v_oo])
            T.writes(Conv[v_n, v_h, v_w, v_o, v_nn, v_oo])
            with T.init():
                Conv[v_n, v_h, v_w, v_o, v_nn, v_oo] = T.float16(0)
            Conv[v_n, v_h, v_w, v_o, v_nn, v_oo] = Conv[v_n, v_h, v_w, v_o, v_nn, v_oo] + Apad[v_n, v_h + v_kh, v_w + v_kw, v_ic, v_nn, v_ii] * W[v_kh, v_kw, v_ic, v_o, v_ii, v_oo]
