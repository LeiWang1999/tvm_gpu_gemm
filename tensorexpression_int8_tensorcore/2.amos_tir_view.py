import tvm
from tvm.script import tir as T
import numpy as np
import os
from tvm.tir.tensor_intrin.cuda import (
    WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN,
    WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN,
    WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN,
    WMMA_LOAD_16x16x16_S8_A_INTRIN,
    WMMA_FILL_16x16x16_S32_INTRIN,
)


log_path = "progress/tensorexpression_int8_tensorcore/2.amos_tir_view"
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

M = 16384
N = 16384
K = 16384
wmma_m = 16
wmma_n = 16
wmma_k = 16
warp_size = 32
block_row_warps = 2
block_col_warps = 2
warp_row_tiles = 2
warp_col_tiles = 8
chunk = 4
vec = 16

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="int8")
        B = T.match_buffer(b, [N, K], dtype="int8")
        C = T.match_buffer(c, [M, N], dtype="int32")

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.int32(0)
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("int32") * B[vj, vk].astype("int32")


ir_module = MyModule
print(ir_module)
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
write_sch(sch, log_path, "original")

block_b = sch.get_block("B")

block_A_tricky = sch.cache_read(block_b, 0, "global")
block_A_tricky_shared = sch.cache_read(block_b, 0, "shared")
block_B_tricky = sch.cache_read(block_b, 1, "global")
block_B_tricky_shared = sch.cache_read(block_b, 1, "shared")
block_C_tricky = sch.cache_write(block_b, 0, "global")

write_sch(sch, log_path, "cache_related")

def schedule_tricky_global(block):
    i, j = sch.get_loops(block)
    i, mi = sch.split(i, [None, 16])
    j, mj = sch.split(j, [None, 16])
    sch.reorder(i, j, mi, mj)
    return (i, j, mi, mj)


block_A_tricky_loops = schedule_tricky_global(block_A_tricky)
block_B_tricky_loops = schedule_tricky_global(block_B_tricky)

write_sch(sch, log_path, "schedule_tricky_global")



# because wmma_m = 16, wmma_n = 16, wmma_k = 16, so we share the same tricky permutation
def tricky_permutation(i, j):
    return (i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n)

sch.transform_layout(block_A_tricky, ("write", 0), tricky_permutation)
sch.transform_layout(block_B_tricky, ("write", 0), tricky_permutation)
sch.transform_layout(block_A_tricky_shared, ("write", 0), tricky_permutation)
sch.transform_layout(block_B_tricky_shared, ("write", 0), tricky_permutation)
sch.transform_layout(block_b, ("write", 0), tricky_permutation)

write_sch(sch, log_path, "permutation")


# schedule block_b

(i, j, k) = sch.get_loops(block_b)
i, kernel_i = sch.split(i, [None, 16])
j, kernel_j = sch.split(j, [None, 16])
k, kernel_k = sch.split(k, [None, 16])
block_i, i, ii = sch.split(i, [None, block_row_warps, warp_row_tiles])
block_j, j, jj = sch.split(j, [None, block_col_warps, warp_col_tiles])
sch.reorder(block_i, block_j, i, j, ii, jj, kernel_i, kernel_j)
sch.bind(block_i, "blockIdx.x")
sch.bind(block_j, "blockIdx.y")
sch.bind(i, "threadIdx.y")
sch.bind(j, "threadIdx.z")

k_o, k_i = sch.split(kernel_k, [None, chunk])
sch.reorder(k_o, k_i, i, j)

write_sch(sch, log_path, "schedule_block_b")

def schedule_shared(block):
    sch.compute_at(block, k_o)
    vector_size = 16
    fused = sch.fuse(*sch.get_loops(block)[-2:])
    _, f_0, f_1, f_2, f_3 = sch.split(
        fused, factors=[None, block_row_warps, block_col_warps, warp_size, vector_size])
    sch.bind(f_2, "threadIdx.x")
    sch.bind(f_1, "threadIdx.y")
    sch.bind(f_0, "threadIdx.z")

schedule_shared(block_A_tricky_shared)
schedule_shared(block_B_tricky_shared)

write_sch(sch, log_path, "schedule_shared_read")









