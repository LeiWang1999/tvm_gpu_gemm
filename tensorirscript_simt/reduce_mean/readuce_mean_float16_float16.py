import tvm.testing
import numpy as np
from tvm import te
import tvm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/tensorirscript_simt/" + fname

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


VERIFY = True

# The sizes of inputs and filters
batch_size = 128
height = 28
width = 28
in_channels = 128
in_dtype = "float16"
out_dtype = "float16"

def mean(N, C, H, W, in_dtype, out_dtype):
    A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="A")

    B = tvm.te.compute(
        [C, N * H * W], lambda i, j: A[j // (H * W), i, j % (H * W) // W, j % W], name="B"
    )

    C_ = tvm.te.compute(
        [N * H * W], lambda i: tvm.tir.const(1.0 / (N * H * W), in_dtype), name="C"
    )

    rk = tvm.te.reduce_axis([0, N * H * W], name="k")
    D = tvm.te.compute(
        [C], lambda i: tvm.te.sum((B[i, rk] * C_[rk]).astype(out_dtype), axis=rk), name="D"
    )

    return [A, B, D]


A, B, Mean = mean(batch_size, in_channels, height, width, in_dtype, out_dtype)

ir_module = te.create_prim_func([A, Mean])
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_b = sch.get_block("B")
block_c = sch.get_block("C")
block_d = sch.get_block("D")

sch.compute_inline(block_b)
sch.compute_inline(block_c)

write_sch(sch, log_path, "compute_inline")
