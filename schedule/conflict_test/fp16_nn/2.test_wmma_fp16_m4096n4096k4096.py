# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
from tvm import te
import numpy as np
from tvm.topi.testing import conv2d_nhwc_python
import tvm.testing
import os

VERIFY = False


def write_code(code, path, fname):
    # if path not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)

def intrin_wmma_load_matrix(shape, scope):
    n, m, l = shape
    if scope == "wmma.matrix_a":
        row, col = n, l
    elif scope == "wmma.matrix_b":
        row, col = l, m
    A = te.placeholder((row, col), name="A", dtype="float16")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=row * col
    )
    C = te.compute((row, col), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=row * col
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                m,
                l,
                BC.elem_offset // (row * col),
                BA.access_ptr("r"),
                8,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(shape):
    n, m, l = shape
    A = te.placeholder((n, l), name="A", dtype="float16")
    B = te.placeholder((l, m), name="B", dtype="float16")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute(
        (n, m),
        lambda ii, jj: te.sum(A[ii, k].astype(
            "float") * B[k, jj].astype("float"), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", data_alignment=32, offset_factor=n * l
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", data_alignment=32, offset_factor=l * m
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        data_alignment=32,
        offset_factor=n * m,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    n,
                    m,
                    l,
                    BC.elem_offset // (n * m),
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // (n * m),
                    BA.data,
                    BA.elem_offset // (n * l),
                    BB.data,
                    BB.elem_offset // (l * m),
                    BC.data,
                    BC.elem_offset // (n * m),
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix(shape):
    n, m, l = shape
    A = te.placeholder((n, m), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=n * m
    )
    C = te.compute((n, m), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="global", data_alignment=32, offset_factor=n * m
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                n,
                m,
                l,
                BA.elem_offset // (n * m),
                BC.access_ptr("w"),
                m,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


@tvm.testing.requires_tensorcore
def test_tensor_core_matmal():
    log_path = "progress/tensorexpression_half_tensorcore/1.tricky_tensorize"
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
    m = 4096
    n = 4096
    l = 4096

    assert n % wmma_m == 0
    assert m % wmma_k == 0
    assert l % wmma_n == 0
    nn, mm, ll = n // wmma_m, m // wmma_n, l // wmma_k
    A = te.placeholder((nn, ll, wmma_m, wmma_k), name="A", dtype="float16")
    B = te.placeholder((ll, mm, wmma_k, wmma_n), name="B", dtype="float16")
    k1 = te.reduce_axis((0, ll), name="k1")
    k2 = te.reduce_axis((0, wmma_k), name="k2")
    C = te.compute(
        (nn, mm, wmma_m, wmma_n),
        lambda i, j, ii, jj: te.sum(
            A[i, k1, ii, k2].astype("float") * B[k1, j, k2, jj].astype("float"), axis=[k1, k2]
        ),
        name="C",
    )
    s = te.create_schedule(C.op)

    warp_size = 32
    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 4
    warp_col_tiles = 4
    chunk = 2
    vec = 8
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")

    i, j, kernel_i, kernel_j = s[C].op.axis
    i, ii = s[C].split(i, factor=warp_row_tiles)
    block_i, i = s[C].split(i, factor=block_row_warps)
    j, jj = s[C].split(j, factor=warp_col_tiles)
    block_j, j = s[C].split(j, factor=block_col_warps)
    s[C].reorder(block_i, block_j, i, j, ii, jj, kernel_i, kernel_j)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(i, thread_y)
    s[C].bind(j, thread_z)

    s[CF].compute_at(s[C], j)
    warp_i, warp_j, _i, _j = s[CF].op.axis
    k, _k = CF.op.reduce_axis
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _i, _j, _k)
    
    s[AF].compute_at(s[CF], ki)
    s[BF].compute_at(s[CF], ki)

    s[AS].compute_at(s[CF], ko)
    # write_code(
    #     str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "compute_at_as.cu")
    xo, yo, xi, yi = AS.op.axis
    t = s[AS].fuse(xo, yo)
    tx, t = s[AS].split(t, nparts=block_row_warps)
    ty, t = s[AS].split(t, nparts=block_col_warps)
    t = s[AS].fuse(xi, yi)
    to, ti = s[AS].split(t, nparts=warp_size)
    ti, vi = s[AS].split(ti, factor=vec)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(to, thread_x)
    s[AS].vectorize(vi)

    s[BS].compute_at(s[CF], ko)
    xo, yo, xi, yi = BS.op.axis
    t = s[BS].fuse(xo, yo)
    tx, t = s[BS].split(t, nparts=block_row_warps)
    ty, t = s[BS].split(t, nparts=block_col_warps)
    t = s[BS].fuse(xi, yi)
    to, ti = s[BS].split(t, nparts=warp_size)
    ti, vi = s[BS].split(ti, factor=vec)
    s[BS].bind(tx, thread_y)
    s[BS].bind(ty, thread_z)
    s[BS].bind(to, thread_x)
    s[BS].vectorize(vi)

    s[AF].tensorize(
        AF.op.axis[-2], intrin_wmma_load_matrix((wmma_m, wmma_n, wmma_k), "wmma.matrix_a"))
    s[BF].tensorize(
        BF.op.axis[-2], intrin_wmma_load_matrix((wmma_m, wmma_n, wmma_k), "wmma.matrix_b"))
    s[CF].tensorize(_i, intrin_wmma_gemm((wmma_m, wmma_n, wmma_k)))
    s[C].tensorize(kernel_i, intrin_wmma_store_matrix(
        (wmma_m, wmma_n, wmma_k)))
        
    func = tvm.build(s, [A, B, C], "cuda")
    write_code(func.imported_modules[0].get_source(), log_path, "tmp.cu")

    dev = tvm.cuda(0)
    a_np = np.random.uniform(size=(nn, ll, wmma_m, wmma_k)).astype(A.dtype)
    b_np = np.random.uniform(size=(ll, mm, wmma_k, wmma_n)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((nn, mm, wmma_m, wmma_n), dtype=C.dtype), dev)
    func(a, b, c)
    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    print("gemm with tensor core: %f ms" % (evaluator(a, b, c).mean * 1e3))

    if VERIFY:
        func(a, b, c)
        a_np = a_np.transpose((0, 1, 3, 2)).reshape(n, n)
        b_np = b_np.transpose((0, 1, 3, 2)).reshape(n, n)
        c_np = c.numpy().transpose((0, 1, 3, 2)).reshape(n, n)
        np.testing.assert_allclose(
            c_np, np.matmul(a_np.astype(C.dtype), b_np.astype(C.dtype)), rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
    test_tensor_core_matmal()
