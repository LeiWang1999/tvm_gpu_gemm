import tvm
from tvm import te
import numpy as np
import tvm.testing
import os
from tvm.tir.buffer import Buffer


def intrin_wmma_load_matrix(shape, scope):
    n, m, l = shape
    if scope == "wmma.matrix_a":
        row, col = n, l
    elif scope == "wmma.matrix_b":
        row, col = m, l
    A = te.placeholder((row, col), name="A", dtype="int8")
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
                col,
                "row_major" if scope == "wmma.matrix_a" else "col_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(shape):
    n, m, l = shape
    A = te.placeholder((n, l), name="A", dtype="int8")
    B = te.placeholder((m, l), name="B", dtype="int8")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute(
        (n, m),
        lambda ii, jj: te.sum(A[ii, k].astype(
            "int32") * B[jj, k].astype("int32"), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", data_alignment=32, offset_factor=n * l
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", data_alignment=32, offset_factor=m * l
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
    A = te.placeholder((n, m), name="A", dtype="int32")
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



def test_gemm():
    log_path = "progress/tensorexpression_int8_tensorcore/2.amos"

    # define wmma
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
    # graph
    nn = 16384
    n = te.var("n")
    n = tvm.runtime.convert(nn)
    m = k = n = nn
    A = te.placeholder((m, k), dtype="int8", name="A")
    B = te.placeholder((n, k), dtype="int8", name="B")
    rk = te.reduce_axis((0, k), name="rk")
    C = te.compute((m, n), lambda ii, jj: te.sum(
        A[ii, rk].astype("int32") * B[jj, rk].astype("int32"), axis=rk), name="C")

    # schedule
    s = te.create_schedule(C.op)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "initial.cu")
    AB = s.cache_read(A, "global", [C])
    BB = s.cache_read(B, "global", [C])
    # AS = s.cache_read(AB, "shared", [C])
    # BS = s.cache_read(BB, "shared", [C])
    # AF = s.cache_read(AS, "local", [C])
    # BF = s.cache_read(BS, "local", [C])
    CF = s.cache_write(C, "global")

    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "cache.cu")

    warp_size = 32
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 2
    warp_col_tiles = 8
    chunk = 4
    vec = 16
    offset = 0
    offsetCS = 0

    def transform_buffer_layer(buffer_stage, inner_shape):
        (m, n) = inner_shape
        i, j, wi, wj = s[buffer_stage].transform_layout(
            lambda i, j: [i // m, j // n, i % m, j % n])
        return (i, j, wi, wj)

    
    AB_axis = transform_buffer_layer(AB, (wmma_m, wmma_k))
    BB_axis = transform_buffer_layer(BB, (wmma_n, wmma_k))

    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "transform_global_input.cu")
    AS_axis = transform_buffer_layer(AS, (wmma_m, wmma_k))
    BS_axis = transform_buffer_layer(BS, (wmma_n, wmma_k))
    
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "transform_shared_input.cu")
    
    AF_axis = transform_buffer_layer(AF, (wmma_m, wmma_k))
    BF_axis = transform_buffer_layer(BF, (wmma_n, wmma_k))
        
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "transform_fragment.cu")

    CF_axis = transform_buffer_layer(CF, (wmma_m, wmma_n))
    k,  = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "transform_intermidiate.cu")

    CB_axis = transform_buffer_layer(CB, (wmma_m, wmma_n))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "transform_global_output.cu")


    # schedule buffer
    def create_kernel_for_global_input_buffer(buffer_stage, axis):
        buffer_block_x = te.thread_axis("blockIdx.x")
        buffer_block_y = te.thread_axis("blockIdx.y")
        buffer_thread_x = te.thread_axis("threadIdx.x")
        buffer_thread_y = te.thread_axis("threadIdx.y")
        buffer_thread_z = te.thread_axis("threadIdx.z") 
        i, j, wi, wj = axis
        t = s[buffer_stage].fuse(wi, wj)
        _, vi = s[buffer_stage].split(t, factor=vec)
        ty, tx = s[buffer_stage].split(j, factor=warp_size)
        s[buffer_stage].bind(tx, buffer_thread_x)
        s[buffer_stage].bind(ty, buffer_thread_y)
        s[buffer_stage].bind(i, buffer_block_x)
        s[buffer_stage].vectorize(vi)

    # create_kernel_for_global_input_buffer(AB, AB_axis)
    # write_code(
    #     str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "7.create_kernel_for_global_input_buffer_A.cu")
    # create_kernel_for_global_input_buffer(BB, BB_axis)
    # write_code(
    #     str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "8.create_kernel_for_global_input_buffer_B.cu")

    def create_kernel_for_shared_output_buffer(buffer_stage):
        buffer_block_x = te.thread_axis("blockIdx.x")
        buffer_block_y = te.thread_axis("blockIdx.y")
        buffer_thread_x = te.thread_axis("threadIdx.x")
        buffer_thread_y = te.thread_axis("threadIdx.y")
        buffer_thread_z = te.thread_axis("threadIdx.z") 
        i, j = buffer_stage.op.axis
        t = s[buffer_stage].fuse(i, j)
        t, vi = s[buffer_stage].split(t, factor=vec)
        t, _ = s[buffer_stage].split(t, factor=16)
        t, tx = s[buffer_stage].split(t, factor=warp_size)
        t, ty = s[buffer_stage].split(t, factor=32)
        s[buffer_stage].bind(tx, buffer_thread_x)
        s[buffer_stage].bind(ty, buffer_thread_y)
        s[buffer_stage].bind(t, buffer_block_x)
        s[buffer_stage].vectorize(vi)
    
    # create_kernel_for_shared_output_buffer(C)
    # write_code(
    #     str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "create_kernel_for_shared_output_buffer.cu")
    
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    
    i, j, kernel_i, kernel_j = CB_axis
    # i, ii = s[CB].split(i, factor=warp_row_tiles)
    # block_i, i = s[CB].split(i, factor=block_row_warps)
    # j, jj = s[CB].split(j, factor=warp_col_tiles)
    # block_j, j = s[CB].split(j, factor=block_col_warps)
    # s[CB].reorder(block_i, block_j, i, j, ii, jj, kernel_i, kernel_j)

    # i, j, = CB.op.axis
    # i, ii = s[CB].split(i, factor=warp_row_tiles)
    # block_i, i = s[CB].split(i, factor=block_row_warps)
    # j, jj = s[CB].split(j, factor=warp_col_tiles)
    # block_j, j = s[CB].split(j, factor=block_col_warps)
    # s[CB].reorder(block_i, block_j, i, j, ii, jj)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "before_schedule_CB.cu")

    s[CB].bind(i, te.thread_axis("blockIdx.x"))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), log_path, "schedule_CB.cu")
    s[CB].bind(j, block_y)
    # s[CB].bind(i, thread_y)
    # s[CB].bind(j, thread_z)


    s[CF].compute_at(s[CB], j)
    warp_i, warp_j, _i, _j = CF_axis
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _i, _j, _k)
    
    s[AF].compute_at(s[CF], ki)
    s[BF].compute_at(s[CF], ki)

    s[AS].compute_at(s[CF], ko)
    xo, yo, xi, yi = AS_axis
    t = s[AS].fuse(xo, yo, xi, yi)
    tx, t = s[AS].split(t, nparts=block_row_warps)
    ty, t = s[AS].split(t, nparts=block_col_warps)
    ti, to = s[AS].split(t, factor=warp_size * vec)
    to, vi = s[AS].split(to, factor=vec)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(to, thread_x)
    s[AS].vectorize(vi)

    s[BS].compute_at(s[CF], ko)
    xo, yo, xi, yi = BS_axis
    t = s[BS].fuse(xo, yo, xi, yi)
    tx, t = s[BS].split(t, nparts=block_row_warps)
    ty, t = s[BS].split(t, nparts=block_col_warps)
    ti, to = s[BS].split(t, factor=warp_size * vec)
    to, vi = s[BS].split(to, factor=vec)
    s[BS].bind(tx, thread_y)
    s[BS].bind(ty, thread_z)
    s[BS].bind(to, thread_x)
    s[BS].vectorize(vi)


    print('-b')
    s[CF].tensorize(_i, intrin_wmma_gemm((wmma_m, wmma_n, wmma_k)))
    s[C].tensorize(kernel_i, intrin_wmma_store_matrix(
        (wmma_m, wmma_n, wmma_k)))
    device = 'cuda -arch=sm_80'

    dev = tvm.device(device, 0)
    # if not dev.exist:
    #     print("Skip because %s is not enabled" % device)
    #     return
    # print("Device %s" % device)
    f = tvm.build(s, [A, B, C], device)
    write_code(f.imported_modules[0].get_source(), "tmp.cu")
    # launch the kernel.
    n, m, l = nn, nn, nn
    a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
    b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), dev)
    for i in range(2):
        f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), np.dot(b_np.T, a_np), rtol=1e1)

    num_flops = 2 * nn * nn * nn
    num_runs = 3
    timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))


if __name__ == "__main__":
    test_gemm()
