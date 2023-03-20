# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(128, 42, 42, 1024), "float16"], W: T.Buffer[(384, 1, 1, 1024), "float16"], Conv: T.Buffer[(225792, 384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    data_im2col_shared = T.alloc_buffer([225792, 1024], dtype="float16", scope="shared")
    data_im2col_shared_warp = T.alloc_buffer([225792, 1024], dtype="float16", scope="warp")
    weight_flatten_shared = T.alloc_buffer([384, 1024], dtype="float16", scope="shared")
    weight_flatten_shared_warp = T.alloc_buffer([384, 1024], dtype="float16", scope="warp")
    Conv_warp = T.alloc_buffer([225792, 384], dtype="float16", scope="warp")
    for ax0, ax1 in T.grid(225792, 1024):
        with T.block("data_im2col_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0 // 1764, v1 // 1024 + v0 % 1764 // 42, v0 % 42, v1 % 1024])
            T.writes(data_im2col_shared[v0, v1])
            data_im2col_shared[v0, v1] = T.if_then_else(0 <= 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) and 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) < 42 and 0 <= 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) and 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) < 42, A[v0 // 1764, 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) - 0, 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) - 0, v1 % 1024], T.float16(0), dtype="float16")
    for ax0, ax1 in T.grid(225792, 1024):
        with T.block("data_im2col_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(data_im2col_shared[v0, v1])
            T.writes(data_im2col_shared_warp[v0, v1])
            data_im2col_shared_warp[v0, v1] = data_im2col_shared[v0, v1]
    for ax0, ax1 in T.grid(384, 1024):
        with T.block("weight_flatten_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(W[v0, v1 // 1024, 0, v1 % 1024])
            T.writes(weight_flatten_shared[v0, v1])
            weight_flatten_shared[v0, v1] = W[v0, v1 // 1024 // 1, v1 // 1024 % 1, v1 % 1024]
    for ax0, ax1 in T.grid(384, 1024):
        with T.block("weight_flatten_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(weight_flatten_shared[v0, v1])
            T.writes(weight_flatten_shared_warp[v0, v1])
            weight_flatten_shared_warp[v0, v1] = weight_flatten_shared[v0, v1]
    for x, y, k in T.grid(225792, 384, 1024):
        with T.block("Conv"):
            v_x, v_y, v_k = T.axis.remap("SSR", [x, y, k])
            T.reads(data_im2col_shared_warp[v_x, v_k], weight_flatten_shared_warp[v_y, v_k])
            T.writes(Conv_warp[v_x, v_y])
            with T.init():
                Conv_warp[v_x, v_y] = T.float16(0)
            Conv_warp[v_x, v_y] = Conv_warp[v_x, v_y] + data_im2col_shared_warp[v_x, v_k] * weight_flatten_shared_warp[v_y, v_k]
    for ax0, ax1 in T.grid(225792, 384):
        with T.block("Conv_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(Conv_warp[v0, v1])
            T.writes(Conv[v0, v1])
            Conv[v0, v1] = Conv_warp[v0, v1]
