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
    Apad_shared = T.allocate([16777216], "float16", "shared")
    Apad_shared_1 = T.buffer_decl([16777216], dtype="float16", data=Apad_shared, scope="shared")
    Apad_shared_2 = T.buffer_decl([1179648], dtype="float16", data=Apad_shared, scope="shared")
    Apad_shared_wmma_matrix_a = T.allocate([16777216], "float16", "wmma.matrix_a")
    Apad_shared_wmma_matrix_a_1 = T.buffer_decl([16777216], dtype="float16", data=Apad_shared_wmma_matrix_a, scope="wmma.matrix_a")
    W_shared_wmma_matrix_b = T.allocate([1179648], "float16", "wmma.matrix_b")
    W_shared_wmma_matrix_b_1 = T.buffer_decl([1179648], dtype="float16", data=W_shared_wmma_matrix_b, scope="wmma.matrix_b")
    Conv_wmma_accumulator = T.allocate([25690112], "float32", "wmma.accumulator")
    Conv_wmma_accumulator_1 = T.buffer_decl([25690112], dtype="float32", data=Conv_wmma_accumulator, scope="wmma.accumulator")
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(16, 16, 16, 16, 16, 16):
        cse_var_3: T.int32 = ax2 * 4096
        cse_var_2: T.int32 = ax3 * 256
        cse_var_1: T.int32 = ax4 * 16
        Apad_shared_1[ax0 * 1048576 + ax1 * 65536 + cse_var_3 + cse_var_2 + cse_var_1 + ax5] = T.if_then_else(1 <= ax1 and ax1 < 15 and 1 <= ax2 and ax2 < 15, A_1[ax0 * 802816 + ax1 * 57344 + cse_var_3 + cse_var_2 + cse_var_1 + ax5 - 61440], T.float16(0), dtype="float16")
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(16, 16, 16, 16, 16, 16):
        cse_var_4: T.int32 = ax0 * 1048576 + ax1 * 65536 + ax2 * 4096 + ax3 * 256 + ax4 * 16 + ax5
        Apad_shared_wmma_matrix_a_1[cse_var_4] = Apad_shared_1[cse_var_4]
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(3, 3, 16, 32, 16, 16):
        cse_var_5: T.int32 = ax0 * 393216 + ax1 * 131072 + ax2 * 8192 + ax3 * 256 + ax4 * 16 + ax5
        Apad_shared_2[cse_var_5] = W_1[cse_var_5]
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(3, 3, 16, 32, 16, 16):
        cse_var_6: T.int32 = ax0 * 393216 + ax1 * 131072 + ax2 * 8192 + ax3 * 256 + ax4 * 16 + ax5
        W_shared_wmma_matrix_b_1[cse_var_6] = Apad_shared_2[cse_var_6]
    for n_c, h_c, w_c, o_c, nn_c, oo_c in T.grid(16, 14, 14, 32, 16, 16):
        Conv_wmma_accumulator_1[n_c * 1605632 + h_c * 114688 + w_c * 8192 + o_c * 256 + nn_c * 16 + oo_c] = T.float32(0)
        for ic, kh, kw, ii in T.grid(16, 3, 3, 16):
            cse_var_9: T.int32 = o_c * 256
            cse_var_8: T.int32 = nn_c * 16
            cse_var_7: T.int32 = n_c * 1605632 + h_c * 114688 + w_c * 8192 + cse_var_9 + cse_var_8 + oo_c
            Conv_wmma_accumulator_1[cse_var_7] = Conv_wmma_accumulator_1[cse_var_7] + T.Cast("float32", Apad_shared_wmma_matrix_a_1[n_c * 1048576 + h_c * 65536 + kh * 65536 + w_c * 4096 + kw * 4096 + ic * 256 + cse_var_8 + ii]) * T.Cast("float32", W_shared_wmma_matrix_b_1[kh * 393216 + kw * 131072 + ic * 8192 + cse_var_9 + ii * 16 + oo_c])
    for n, h, w, o, nn, oo in T.grid(16, 14, 14, 32, 16, 16):
        cse_var_10: T.int32 = n * 1605632 + h * 114688 + w * 8192 + o * 256 + nn * 16 + oo
        Conv_1[cse_var_10] = Conv_wmma_accumulator_1[cse_var_10]
