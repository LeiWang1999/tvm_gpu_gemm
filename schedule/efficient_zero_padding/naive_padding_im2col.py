import tvm
from tvm.script import tir as T

# The sizes of inputs and filters
batch_size = 128
height = 28
width = 28
in_channels = 128
out_channels = 128
kernel_h = 3
kernel_w = 3
pad_h = 1
pad_w = 1
stride_h = 1
stride_w = 1
dilation_h = 1
dilation_w = 1
output_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
output_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
print("output_height: ", output_height)
print("output_width: ", output_width)
M = batch_size * output_height * output_width
N = out_channels
K = kernel_h * kernel_w * in_channels
# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size,
    height,
    width,
    in_channels
)

# Algorithm
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, data: T.handle,):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        A = T.match_buffer(a, (batch_size, height, width, in_channels), dtype="float16")
        Apad = T.alloc_buffer([batch_size, height + 2*pad_h, width + 2*pad_w, in_channels], dtype="float16")
        data_im2col = T.match_buffer(data, [M, K], dtype="float16")

        for n, h, w, i in T.grid(batch_size, height + 2*pad_h, width + 2*pad_w, in_channels):
            with T.block("Apad"):
                v_n, v_h, v_w, v_i = T.axis.remap("SSSS", [n, h, w, i])
                Apad[v_n, v_h, v_w, v_i] = T.if_then_else(pad_h <= v_h and v_h < height + pad_h and pad_w <= v_w and v_w < width + pad_w, A[v_n, v_h - pad_h, v_w - pad_w, v_i], T.float16(0), dtype="float16")
        for x, y in T.grid(M, K):
            with T.block("data_im2col"):
                v_x, v_y = T.axis.remap("SS", [x, y])
                data_im2col[v_x, v_y] = Apad[
                v_x // (output_height * output_width),
                stride_h * ((v_x % (output_height * output_width)) // output_width) + dilation_h * ((v_y // in_channels) // kernel_w),
                stride_w * ((v_x % (output_height * output_width)) % output_width) + dilation_w * ((v_y // in_channels) % kernel_w),
                v_y % in_channels,
            ]
"""
efficient res:
  for (int ax0_ax1_fused_0_cache = 0; ax0_ax1_fused_0_cache < 2; ++ax0_ax1_fused_0_cache) {
    PadInput_reindex_shared_dyn_local[ax0_ax1_fused_0_cache] = (((28 <= ((((((((int)blockIdx.y) * 256) + (((int)blockIdx.x) * 128)) + (ax0_ax1_fused_0_cache * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) % 784)) && (1 <= ((((((((int)blockIdx.y) * 256) + (((int)blockIdx.x) * 128)) + (ax0_ax1_fused_0_cache * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) % 28))) ? *(uint4*)(inputs + (((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 16384)) + (ax0_ax1_fused_0_cache * 8192)) + (((int)threadIdx.y) * 1024)) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 8)) - 3712)) : make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f))));
  }
"""

ir_module = MyModule
sch = tvm.tir.Schedule(ir_module, debug_mask="all")

block_pad = sch.get_block("Apad")
block_im2col = sch.get_block("data_im2col")
sch.compute_inline(block_pad)

i, j = sch.get_loops(block_im2col)
j, ci = sch.split(j, factors=[None, 8])

sch.bind(i, "blockIdx.x")
sch.bind(j, "threadIdx.x")

sch.vectorize(ci)

ctx = tvm.cuda(1)
cuda_mod = tvm.build(sch.mod, target="cuda")
print(cuda_mod.imported_modules[0].get_source())