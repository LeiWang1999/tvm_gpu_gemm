"""
    Considering a gemm problem, in this part we try to leverage the ldmatrix, mma, and stmatrix to do the computation.
    The ldmatrix and stmatrix are used to load and store the data from global memory to shared memory.
    The mma is used to do the computation.
    thread_x will be set into 32, which represents the number of threads in a warp.
    thread_y and thread_z will be set into value which represents the array of warps. 
"""
import tvm
from tvm.script import tir as T
from tvm import te, tir, topi
import numpy as np
import os

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/tensorirscript_imma/" + fname
count = 0
def write_code(code, path, fname):
    global count
    # if path not exist, create it
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)

def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)


# The sizes of inputs and filters
batch_size = 128
height = 30
width = 30
in_channels = 256
out_channels = 256
kernel_h = 3
kernel_w = 3
pad_h = 0
pad_w = 0
stride_h = 2
stride_w = 2
dilation_h = 1
dilation_w = 1
output_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
output_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
print("output_height: ", output_height)
print("output_width: ", output_width)
# TensorCore shape
wmma_m = 16
wmma_n = 16
wmma_k = 16

# tuning params
warp_size = 32
block_row_warps = 1
block_col_warps = 1
warp_row_tiles = 1
warp_col_tiles = 2
chunk = 1
vec = 8
wmma_m = 16
wmma_n = 16
wmma_k = 16
raster = 1
stage = 9

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size,
    height,
    width,
    in_channels
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    kernel_h,
    kernel_w,
    in_channels,
    out_channels
)
# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size,
    output_height,
    output_width,
    out_channels,
)
M = batch_size * output_height * output_width
N = out_channels
K = kernel_h * kernel_w * in_channels

print("M: ", M)
print("N: ", N)
print("K: ", K)
# padding MPAD as the multiple of block_row_warps * warp_row_tiles * wmma_m
MPAD = (M + block_row_warps * warp_row_tiles * wmma_m - 1) // (
    block_row_warps * warp_row_tiles * wmma_m
) * block_row_warps * warp_row_tiles * wmma_m
# padding NPAD as the multiple of block_col_warps * warp_col_tiles * wmma_n
NPAD = (N + block_col_warps * warp_col_tiles * wmma_n - 1) // (
    block_col_warps * warp_col_tiles * wmma_n
) * block_col_warps * warp_col_tiles * wmma_n
# padding KPAD as the multiple of block_col_warps * warp_col_tiles * wmma_k
KPAD = (K + block_col_warps * warp_col_tiles * wmma_k - 1) // (
    block_col_warps * warp_col_tiles * wmma_k
) * block_col_warps * warp_col_tiles * wmma_k

print("MPAD: ", MPAD)
print("NPAD: ", NPAD)
print("KPAD: ", KPAD)

