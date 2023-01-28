# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1, 56, 56, 64), "float16"], W: T.Buffer[(3, 3, 64, 64), "float16"], Conv: T.Buffer[(1, 3136, 64), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    data_im2colPad_shared = T.alloc_buffer([1, 200, 40, 16, 16], dtype="float16", scope="shared")
    data_im2colPad_shared_wmma_matrix_a = T.alloc_buffer([1, 200, 40, 16, 16], dtype="float16", scope="wmma.matrix_a")
    weight_flattenPad_shared = T.alloc_buffer([40, 8, 16, 16], dtype="float16", scope="shared")
    weight_flattenPad_shared_wmma_matrix_b = T.alloc_buffer([40, 8, 16, 16], dtype="float16", scope="wmma.matrix_b")
    CPad_shared = T.alloc_buffer([1, 3200, 128], dtype="float16", scope="shared")
    CPad_shared_wmma_accumulator = T.alloc_buffer([1, 200, 8, 16, 16], dtype="float16", scope="wmma.accumulator")
    for ax0, ax1, ax2 in T.grid(1, 3200, 640):
        with T.block("data_im2colPad_shared"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v0, v2 // 192 + v1 // 56 - 1, v2 % 192 // 64 + v1 % 56 - 1, v2 % 64])
            T.writes(data_im2colPad_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            data_im2colPad_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16] = T.if_then_else(v1 < 3136 and v2 < 576, T.if_then_else(1 <= 1 * (v1 // 56) + 1 * (v2 // 64 // 3) and 1 * (v1 // 56) + 1 * (v2 // 64 // 3) < 57 and 1 <= 1 * (v1 % 56) + 1 * (v2 // 64 % 3) and 1 * (v1 % 56) + 1 * (v2 // 64 % 3) < 57, A[v0, 1 * (v1 // 56) + 1 * (v2 // 64 // 3) - 1, 1 * (v1 % 56) + 1 * (v2 // 64 % 3) - 1, v2 % 64], T.float16(0), dtype="float16"), T.float16(0), dtype="float16")
    for ax0, ax1, ax2 in T.grid(1, 3200, 640):
        with T.block("data_im2colPad_shared_wmma.matrix_a"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(data_im2colPad_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            T.writes(data_im2colPad_shared_wmma_matrix_a[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            data_im2colPad_shared_wmma_matrix_a[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16] = data_im2colPad_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16]
    for ax0, ax1 in T.grid(640, 128):
        with T.block("weight_flattenPad_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(W[v0 // 192, v0 % 192 // 64, v0 % 64, v1])
            T.writes(weight_flattenPad_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            weight_flattenPad_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = T.if_then_else(v0 < 576 and v1 < 64, W[v0 // 64 // 3, v0 // 64 % 3, v0 % 64, v1], T.float16(0), dtype="float16")
    for ax0, ax1 in T.grid(640, 128):
        with T.block("weight_flattenPad_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(weight_flattenPad_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(weight_flattenPad_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            weight_flattenPad_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = weight_flattenPad_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for n, x_0, y_0, k_0, x_1, y_1, k_1 in T.grid(1, 200, 8, 40, 16, 16, 16):
        with T.block("Conv"):
            v_n = T.axis.spatial(1, n)
            v_x = T.axis.spatial(3200, x_0 * 16 + x_1)
            v_y = T.axis.spatial(128, y_0 * 16 + y_1)
            v_k = T.axis.reduce(640, k_0 * 16 + k_1)
            T.reads(data_im2colPad_shared_wmma_matrix_a[v_n, v_x // 16, v_k // 16, v_x % 16, v_k % 16], weight_flattenPad_shared_wmma_matrix_b[v_k // 16, v_y // 16, v_k % 16, v_y % 16])
            T.writes(CPad_shared_wmma_accumulator[v_n, v_x // 16, v_y // 16, v_x % 16, v_y % 16])
            with T.init():
                CPad_shared_wmma_accumulator[v_n, v_x // 16, v_y // 16, v_x % 16, v_y % 16] = T.float16(0)
            CPad_shared_wmma_accumulator[v_n, v_x // 16, v_y // 16, v_x % 16, v_y % 16] = CPad_shared_wmma_accumulator[v_n, v_x // 16, v_y // 16, v_x % 16, v_y % 16] + data_im2colPad_shared_wmma_matrix_a[v_n, v_x // 16, v_k // 16, v_x % 16, v_k % 16] * weight_flattenPad_shared_wmma_matrix_b[v_k // 16, v_y // 16, v_k % 16, v_y % 16]
    for ax0, ax1, ax2 in T.grid(1, 3200, 128):
        with T.block("CPad_shared_wmma.accumulator"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(CPad_shared_wmma_accumulator[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            T.writes(CPad_shared[v0, v1, v2])
            CPad_shared[v0, v1, v2] = CPad_shared_wmma_accumulator[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16]
    for ax0, ax1, ax2 in T.grid(1, 3200, 128):
        with T.block("CPad_shared"):
            T.where(ax1 < 3136 and ax2 < 64)
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(CPad_shared[v0, v1, v2])
            T.writes(Conv[v0, v1, v2])
            Conv[v0, v1, v2] = CPad_shared[v0, v1, v2]
