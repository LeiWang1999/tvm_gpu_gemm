import tvm.testing
import numpy as np
from tvm import te
import tvm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from intrin.DP4A_ReduceSum import DP4A_REDUCE_SUM_INTRIN
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
batch_size = 256
height = 32
width = 32
in_channels = 256
in_dtype = "int8"
out_dtype = "int32"
M = in_channels
K = batch_size * height * width

num_warps = 4
chunk = 4

warp_size = 32
vec = 16
MMA_M = 1
MMA_N = 1
MMA_K = 4


if chunk * MMA_K < vec:
    vec = chunk * MMA_K

num_tx = (height * width) // vec
print("num_tx = ", num_tx)
print("num_warps = ", num_warps)


def sum(N, C, H, W, in_dtype, out_dtype):
    A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="A")

    B = tvm.te.compute(
        [C, N * H * W], lambda i, j: A[j // (H * W), i, j % (H * W) // W, j % W], name="B"
    )

    rk = tvm.te.reduce_axis([0, N * H * W], name="k")
    if out_dtype == "int32":
        _C = tvm.te.compute(
            [C], lambda i: tvm.te.sum((B[i, rk]).astype(out_dtype), axis=rk), name="C"
        )
        
    return [A, B, _C]


A, B, Sum = sum(batch_size, in_channels, height, width, in_dtype, out_dtype)

ir_module = te.create_prim_func([A, Sum])
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_b = sch.get_block("B")
block_c = sch.get_block("C")

block_local_A = sch.cache_read(block_c, 0, "local")
block_local_C = sch.cache_write(block_c, 0, "local")
write_sch(sch, log_path, "cache_related")
sch.compute_inline(block_b)
write_sch(sch, log_path, "compute_inline")

i, k = sch.get_loops(block_c)
ko, tz, ki, tx, vk, kernel_k = sch.split(k, factors=[None, num_warps, chunk, num_tx, vec // MMA_K, MMA_K])
bx = i
sch.reorder(bx, ko, tz, ki, tx, kernel_k)

sch.bind(bx, "blockIdx.x")
sch.bind(tz, "threadIdx.z")
# sch.bind(ty, "threadIdx.y")
sch.bind(tx, "threadIdx.x")
write_sch(sch, log_path, "do_split")

# cache read A from global memory to shared_memory
sch.compute_at(block_local_A, tx, preserve_unit_loops=True)
sch.reverse_compute_at(block_local_C, bx, preserve_unit_loops=True)
write_sch(sch, log_path, "compute_at_related")

block_local_a_i, block_local_a_k = sch.get_loops(block_local_A)[-2:]
sch.vectorize(block_local_a_k)
# sch.decompose_reduction(block_c, ko)
write_sch(sch, log_path, "decompose_reduction")

# sch.tensorize(kernel_k, DP4A_REDUCE_SUM_INTRIN)
write_sch(sch, log_path, "tensorize")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), log_path, "tmp.cu")

# a_np = np.ones((batch_size, in_channels, height, width)).astype("int8")
a_np = np.mod(np.arange(M * K), 4).reshape((batch_size, in_channels, height, width)).astype("int8")
b_np = np.zeros((M)).astype("int32")

cuda_a = tvm.nd.array((a_np).astype("int8"), ctx)
cuda_c = tvm.nd.array((b_np).astype("int32"), ctx)
cuda_mod(cuda_a, cuda_c)

if VERIFY:
    import torch
    a_torch = torch.tensor(a_np, dtype=torch.int8)
    b_torch = torch.sum(a_torch, dim=[0, 2, 3], dtype=torch.int32)
    np.testing.assert_allclose(cuda_c.asnumpy(), b_torch.numpy(), rtol=1e-3)
    

num_flops = 2 * M * K
num_runs = 3
timer_cuda_mod = cuda_mod.time_evaluator(
    cuda_mod.entry_name, ctx, number=num_runs)

t = timer_cuda_mod(cuda_a, cuda_c).mean

GFLOPS = num_flops / (t * 1e3) / 1e6
print("average time cost of %d runs = %g ms, %g GFLOPS." %
      (num_runs, t * 1e3, GFLOPS))
