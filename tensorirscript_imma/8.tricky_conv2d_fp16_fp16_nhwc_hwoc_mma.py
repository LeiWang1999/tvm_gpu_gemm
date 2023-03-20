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
from intrin.tricky_mma_float16_float16 import (
    TRICKY_MMA_fill_16x16_f16_INTRIN,
    TRICKY_LDMATRIX_16x16_A_INTRIN,
    TRICKY_LDMATRIX_16x16_B_INTRIN,
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
    TRICKY_MMA_f16f16f16_INTRIN,
    TRICKY_MMA_f16f16f16_TRANS_INTRIN,
    TRICKY_MMA_store_16x16_f16_global_INTRIN,
    A_global_16x16_to_shared_load_16x16_layout,
    C_shared_16x16_to_ldmatrix_32x8_layout,
    A_B_shared_16x16_to_ldmatrix_32x8_layout
)



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
in_channels = 128
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


assert batch_size % wmma_k == 0
assert in_channels % wmma_m == 0
assert out_channels % wmma_n == 0

# tuning params
block_row_warps = 1
block_col_warps = 1
warp_row_tiles = 1
warp_col_tiles = 1
warp_size = 32
chunk = 1

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size // wmma_m,
    height,
    width,
    in_channels // wmma_k,
    wmma_m,
    wmma_k,
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    kernel_h,
    kernel_w,
    in_channels // wmma_k,
    out_channels // wmma_n,
    wmma_k,
    wmma_n,
)
# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size // wmma_m,
    height,
    width,
    out_channels // wmma_n,
    wmma_m,
    wmma_n,
)
# Reduction axes
kh = te.reduce_axis((0, kernel_h), name="kh")
kw = te.reduce_axis((0, kernel_w), name="kw")
ic = te.reduce_axis((0, in_channels // wmma_k), name="ic")
ii = te.reduce_axis((0, wmma_k), name="ii")
# Algorithm
A = te.placeholder(data_shape, name="A", dtype="float16")
W = te.placeholder(kernel_shape, name="W", dtype="float16")
Apad = te.compute(
    (
        batch_size // wmma_m,
        height + 2 * pad_h,
        width + 2 * pad_w,
        in_channels // wmma_k,
        wmma_m,
        wmma_k,
    ),
    lambda n, h, w, i, nn, ii: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height, w >= pad_w, w - pad_w < width),
        A[n, h - pad_h, w - pad_w, i, nn, ii],
        tvm.tir.const(0.0, "float16"),
    ),
    name="Apad",
)
Conv = te.compute(
    output_shape,
    lambda n, h, w, o, nn, oo: te.sum(
        Apad[n, h * stride_h + kh, w * stride_w + kw, ic, nn, ii].astype("float16")
        * W[kh, kw, ic, o, ii, oo].astype("float16"),
        axis=[ic, kh, kw, ii],
    ),
    name="Conv",
)

ir_module = te.create_prim_func([A, W, Conv])

print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_pad = sch.get_block("Apad")
# block_pad_decompose = sch.decompose_padding(block_pad, sch.get_loops(block_pad)[0])
# write_sch(sch, log_path, "decompose_padding")

block_conv = sch.get_block("Conv")
block_conv_input_shared = sch.cache_read(block_conv, 0 ,"shared")
block_conv_input_frag = sch.cache_read(block_conv, 0, "warp")
block_conv_weight_shared = sch.cache_read(block_conv, 1 ,"shared")
block_conv_weight_frag = sch.cache_read(block_conv, 1, "warp")
block_conv_output_frag = sch.cache_write(block_conv, 0, "warp")
write_sch(sch, log_path, "cache_related")

# 128x32
def permutation(l0, l1, i, j, kernel_i, kernel_j):
    return (l0, l1, i, j, *A_global_16x16_to_shared_load_16x16_layout(kernel_i, kernel_j))

sch.compute_inline(block_pad)
write_sch(sch, log_path, "PadInputInline")

sch.transform_layout(block_conv_input_shared, ("read", 0),
                     permutation)
sch.transform_layout(block_conv_weight_shared, ("read", 0),
                     permutation)

write_sch(sch, log_path, "transform_layout")



nc, hc, wc, oc, nnc, ooc, ic, kh, kw, ii = sch.get_loops(block_conv)
block_k = sch.fuse(hc, wc)
sch.bind(block_k, "blockIdx.z")
nc, nci = sch.split(nc, factors=[None, warp_row_tiles])
block_i, nc = sch.split(nc, factors=[None, block_row_warps])
oc, oci = sch.split(oc, factors=[None, warp_col_tiles])
block_j, oc = sch.split(oc, factors=[None, block_col_warps])
sch.reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
sch.bind(block_i, "blockIdx.x")
sch.bind(block_j, "blockIdx.y")
sch.bind(nc,"threadIdx.y")
sch.bind(oc,"threadIdx.z")
write_sch(sch, log_path, "schedule_block_conv")

sch.reverse_compute_at(block_conv_output_frag, oc, preserve_unit_loops=True)
n_1, o_1, nn, oo, ic, kh, kw, ii = sch.get_loops(block_conv)[-8:]
ko, ki = sch.split(ic, factors=[None, chunk])
sch.reorder(ko, kh, ki, kw, n_1, o_1, nn, oo, ii)
write_sch(sch, log_path, "reverse_compute_at_output_frag")

sch.compute_at(block_conv_input_frag, kw, preserve_unit_loops=True)
sch.compute_at(block_conv_weight_frag, kw, preserve_unit_loops=True)
write_sch(sch, log_path, "compute_at_input_frag")

def schedule_shared_A(block):
    sch.compute_at(block, kh, preserve_unit_loops=True)
    i, j = sch.get_loops(block)[-2:]
    t = sch.get_loops(block)[-5]
    tx, xo = sch.split(t , factors=[block_row_warps, None])
    ty, yo = sch.split(xo, factors=[block_col_warps, None])
    t = sch.fuse(i, j)
    to, ti = sch.split(t, factors=[None, warp_size])
    sch.bind(ty, "threadIdx.z")
    sch.bind(tx, "threadIdx.y")
    sch.bind(ti, "threadIdx.x")

def schedule_shared_B(block):
    sch.compute_at(block, kh, preserve_unit_loops=True)
    # t = sch.get_loops(block)[-3]
    t = sch.fuse(*sch.get_loops(block)[-5:-2])
    i, j = sch.get_loops(block)[-2:]
    tx, xo = sch.split(t, factors=[block_row_warps, None])
    ty, yo = sch.split(xo, factors=[block_col_warps, None])
    t = sch.fuse(i, j)
    to, ti = sch.split(t, factors=[None, warp_size])
    sch.bind(ty, "threadIdx.z")
    sch.bind(tx, "threadIdx.y")
    sch.bind(ti, "threadIdx.x")

# schedule the shared memory into A
schedule_shared_A(block_conv_input_shared)
write_sch(sch, log_path, "schedule_A_shared")

# schedule the shared memory into B
schedule_shared_B(block_conv_weight_shared)
write_sch(sch, log_path, "schedule_B_shared")

init_block_conv = sch.decompose_reduction(block_conv, ko)
write_sch(sch, log_path, "decompose_reduction")

def index_map_A(l0, l1, i, k, wmma_m, wmma_k):
    return (l0, l1, i, k, *A_B_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_k))

def index_map_C(l0, l1, i, j, wmma_m, wmma_n):
    return (l0, l1, i, j, *C_shared_16x16_to_ldmatrix_32x8_layout(wmma_m, wmma_n),)


sch.transform_layout(block_conv_input_frag, ("write", 0), index_map_A)
sch.transform_layout(block_conv_weight_frag, ("write", 0), index_map_A)
sch.transform_layout(block_conv_output_frag, ("read", 0), index_map_C)
write_sch(sch, log_path, "transform_layout")

sch.tensorize(sch.get_loops(init_block_conv)[-2], TRICKY_MMA_fill_16x16_f16_INTRIN)
write_sch(sch, log_path, "tensorize_wmma_fill")

sch.tensorize(sch.get_loops(block_conv_input_frag)[-2], TRICKY_LDMATRIX_16x16_A_INTRIN)
sch.tensorize(sch.get_loops(block_conv_weight_frag)[-2], TRICKY_LDMATRIX_16x16_B_INTRIN)
write_sch(sch, log_path, "tensorize_ldmatrix")

sch.tensorize(nn, TRICKY_MMA_f16f16f16_INTRIN)
write_sch(sch, log_path, "tensorize_wmma_sync")

sch.tensorize(sch.get_loops(block_conv_output_frag)[-2], TRICKY_MMA_store_16x16_f16_global_INTRIN)
write_sch(sch, log_path, "tensorize_store")


ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")
