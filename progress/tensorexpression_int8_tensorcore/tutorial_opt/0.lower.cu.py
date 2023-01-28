# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16, 14, 14, 16, 16, 16), "float16"], W: T.Buffer[(3, 3, 16, 32, 16, 16), "float16"], Conv: T.Buffer[(16, 14, 14, 32, 16, 16), "float32"]):
    # function attr dict
    T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
    # buffer definition
    A_1 = T.buffer_decl([12845056], dtype="float16", data=A.data)
    Conv_1 = T.buffer_decl([25690112], dtype="float32", data=Conv.data)
    W_1 = T.buffer_decl([1179648], dtype="float16", data=W.data)
    # body
    for n, h, w, o, nn, oo in T.grid(16, 14, 14, 32, 16, 16):
        Conv_1[n * 1605632 + h * 114688 + w * 8192 + o * 256 + nn * 16 + oo] = T.float32(0)
        for ic, kh, kw, ii in T.grid(16, 3, 3, 16):
            cse_var_5: T.int32 = o * 256
            cse_var_4: T.int32 = nn * 16
            cse_var_3: T.int32 = h + kh
            cse_var_2: T.int32 = w + kw
            cse_var_1: T.int32 = n * 1605632 + h * 114688 + w * 8192 + cse_var_5 + cse_var_4 + oo
            Conv_1[cse_var_1] = Conv_1[cse_var_1] + T.Cast("float32", T.if_then_else(1 <= cse_var_3 and cse_var_3 < 15 and 1 <= cse_var_2 and cse_var_2 < 15, A_1[n * 802816 + h * 57344 + kh * 57344 + w * 4096 + kw * 4096 + ic * 256 + cse_var_4 + ii - 61440], T.float16(0), dtype="float16")) * T.Cast("float32", W_1[kh * 393216 + kw * 131072 + ic * 8192 + cse_var_5 + ii * 16 + oo])
