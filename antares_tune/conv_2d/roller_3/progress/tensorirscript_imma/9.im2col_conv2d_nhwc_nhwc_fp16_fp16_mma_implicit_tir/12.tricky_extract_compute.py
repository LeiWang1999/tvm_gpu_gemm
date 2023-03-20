# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(128, 42, 42, 1024), "float16"], W: T.Buffer[(384, 1, 1, 1024), "float16"], Conv: T.Buffer[(225792, 384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    data_im2col_shared = T.alloc_buffer([14112, 64, 16, 16], dtype="float16", scope="shared")
    data_im2col_shared_warp = T.alloc_buffer([14112, 64, 16, 16], dtype="float16", scope="warp")
    weight_flatten_shared = T.alloc_buffer([24, 64, 16, 16], dtype="float16", scope="shared")
    weight_flatten_shared_warp = T.alloc_buffer([24, 64, 16, 16], dtype="float16", scope="warp")
    Conv_warp = T.alloc_buffer([14112, 24, 16, 16], dtype="float16", scope="warp")
    for ax0, ax1 in T.grid(225792, 1024):
        with T.block("data_im2col_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0 // 1764, v1 // 1024 + v0 % 1764 // 42, v0 % 42, v1 % 1024])
            T.writes(data_im2col_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            data_im2col_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = T.if_then_else(0 <= 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) and 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) < 42 and 0 <= 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) and 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) < 42, A[v0 // 1764, 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) - 0, 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) - 0, v1 % 1024], T.float16(0), dtype="float16")
    for ax0, ax1 in T.grid(225792, 1024):
        with T.block("data_im2col_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(data_im2col_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(data_im2col_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            data_im2col_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = data_im2col_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(384, 1024):
        with T.block("weight_flatten_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(W[v0, v1 // 1024, 0, v1 % 1024])
            T.writes(weight_flatten_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            weight_flatten_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = W[v0, v1 // 1024 // 1, v1 // 1024 % 1, v1 % 1024]
    for ax0, ax1 in T.grid(384, 1024):
        with T.block("weight_flatten_shared_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(weight_flatten_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(weight_flatten_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            weight_flatten_shared_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = weight_flatten_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for x_0, y_0, k_0, x_1, y_1, k_1 in T.grid(14112, 24, 64, 16, 16, 16):
        with T.block("Conv"):
            v_x = T.axis.spatial(225792, x_0 * 16 + x_1)
            v_y = T.axis.spatial(384, y_0 * 16 + y_1)
            v_k = T.axis.reduce(1024, k_0 * 16 + k_1)
            T.reads(data_im2col_shared_warp[v_x // 16, v_k // 16, v_x % 16, v_k % 16], weight_flatten_shared_warp[v_y // 16, v_k // 16, v_y % 16, v_k % 16])
            T.writes(Conv_warp[v_x // 16, v_y // 16, v_x % 16, v_y % 16])
            with T.init():
                Conv_warp[v_x // 16, v_y // 16, v_x % 16, v_y % 16] = T.float16(0)
            Conv_warp[v_x // 16, v_y // 16, v_x % 16, v_y % 16] = Conv_warp[v_x // 16, v_y // 16, v_x % 16, v_y % 16] + data_im2col_shared_warp[v_x // 16, v_k // 16, v_x % 16, v_k % 16] * weight_flatten_shared_warp[v_y // 16, v_k // 16, v_y % 16, v_k % 16]
    for ax0, ax1 in T.grid(225792, 384):
        with T.block("Conv_warp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(Conv_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(Conv[v0, v1])
            Conv[v0, v1] = Conv_warp[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
