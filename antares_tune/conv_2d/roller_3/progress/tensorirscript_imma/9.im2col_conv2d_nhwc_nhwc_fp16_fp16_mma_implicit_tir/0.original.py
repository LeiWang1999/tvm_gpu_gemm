# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(128, 42, 42, 1024), "float16"], W: T.Buffer[(384, 1, 1, 1024), "float16"], Conv: T.Buffer[(225792, 384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    Apad = T.alloc_buffer([128, 42, 42, 1024], dtype="float16")
    data_im2col = T.alloc_buffer([225792, 1024], dtype="float16")
    weight_flatten = T.alloc_buffer([384, 1024], dtype="float16")
    for n, h, w, i in T.grid(128, 42, 42, 1024):
        with T.block("Apad"):
            v_n, v_h, v_w, v_i = T.axis.remap("SSSS", [n, h, w, i])
            T.reads(A[v_n, v_h, v_w, v_i])
            T.writes(Apad[v_n, v_h, v_w, v_i])
            Apad[v_n, v_h, v_w, v_i] = T.if_then_else(0 <= v_h and v_h < 42 and 0 <= v_w and v_w < 42, A[v_n, v_h - 0, v_w - 0, v_i], T.float16(0), dtype="float16")
    for x, y in T.grid(225792, 1024):
        with T.block("data_im2col"):
            v_x, v_y = T.axis.remap("SS", [x, y])
            T.reads(Apad[v_x // 1764, v_y // 1024 + v_x % 1764 // 42, v_x % 42, v_y % 1024])
            T.writes(data_im2col[v_x, v_y])
            data_im2col[v_x, v_y] = Apad[v_x // 1764, 1 * (v_x % 1764 // 42) + 1 * (v_y // 1024 // 1), 1 * (v_x % 1764 % 42) + 1 * (v_y // 1024 % 1), v_y % 1024]
    for x, y in T.grid(384, 1024):
        with T.block("weight_flatten"):
            v_n, v_k = T.axis.remap("SS", [x, y])
            T.reads(W[v_n, v_k // 1024, 0, v_k % 1024])
            T.writes(weight_flatten[v_n, v_k])
            weight_flatten[v_n, v_k] = W[v_n, v_k // 1024 // 1, v_k // 1024 % 1, v_k % 1024]
    for x, y, k in T.grid(225792, 384, 1024):
        with T.block("Conv"):
            v_x, v_y, v_k = T.axis.remap("SSR", [x, y, k])
            T.reads(data_im2col[v_x, v_k], weight_flatten[v_y, v_k])
            T.writes(Conv[v_x, v_y])
            with T.init():
                Conv[v_x, v_y] = T.float16(0)
            Conv[v_x, v_y] = Conv[v_x, v_y] + data_im2col[v_x, v_k] * weight_flatten[v_y, v_k]
