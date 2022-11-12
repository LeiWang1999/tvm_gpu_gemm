import tvm
from tvm import te


def dp4a(x_scope="shared", y_scope="shared", z_scope="local", dtypes=("int8", "int8")):
    """
    Int8 dot product reduced by every 4 elements using __dp4a

    Parameters
    ----------
    x_scope : str, optional
        The storage scope of buffer for lhs
    y_scope : str, optional
        The storage scope of buffer for rhs
    z_scope : str, optional
        The storage scope of buffer for result
    dtypes:  tuple of strs, optional
        The dtype of x and y

    Returns
    -------
    intrin : TensorIntrin
        The dp4a TensorIntrin that can be used in tensorizing schedule.
    """

    n = 4  # dp4a requires operands packed by 4
    result_dtype = "int32" if dtypes[1] == "int8" else "uint32"

    x = te.placeholder((n,), name="x", dtype=dtypes[0])
    y = te.placeholder((n,), name="y", dtype=dtypes[1])

    k = te.reduce_axis((0, n), name="rc")

    z = te.compute(
        (1,), lambda i: te.sum(x[k].astype(result_dtype)
                               * y[k].astype(result_dtype), axis=[k])
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            xx, yy = ins
            zz = outs[0]
            zz_dtype = zz.dtype

            if index == 1:
                return zz.vstore(0, tvm.tir.const(0, zz_dtype))

            ib = tvm.tir.ir_builder.create()

            vec_x_dtype = "int8x4" if xx.dtype == "int8" else "uint8x4"
            vec_y_dtype = "int8x4" if yy.dtype == "int8" else "uint8x4"

            vec_x = xx.vload(0, dtype=vec_x_dtype)
            vec_y = yy.vload(0, dtype=vec_y_dtype)
            prev_z = 0 if index == 0 else zz.vload(0)

            # if is_target("rocm"):
            #     # TODO(masahi): Here we are assuming that we are compiling for gfx10 or later
            #     # We can refine the specification for dot product on rocm if needed later.

            #     # We can just use "llvm.amdgcn.udot4" for u8u8u32, but it is not tested.
            #     assert (
            #         dtypes[0] == "int8" and dtypes[0] == "int8"
            #     ), "u8u8u32 dot product for rocm not supported yet"

            #     new_z = tvm.tir.call_llvm_pure_intrin(
            #         zz_dtype,
            #         "llvm.amdgcn.sdot4",
            #         tvm.tir.const(4, "uint32"),
            #         tvm.tir.call_intrin("int32", "tir.reinterpret", vec_x),
            #         tvm.tir.call_intrin("int32", "tir.reinterpret", vec_y),
            #         prev_z,
            #         True,
            #     )
            # else:
            #     new_z = tvm.tir.call_pure_extern(zz_dtype, "__dp4a", vec_x, vec_y, prev_z)
            new_z = tvm.tir.call_pure_extern(
                zz_dtype, "__dp4a", vec_x, vec_y, prev_z)
            ib.emit(zz.vstore(0, new_z))

            return ib.get()

        return _instr(0), _instr(1), _instr(2)  # body, reset, update

    default_buffer_params = {"data_alignment": 4, "offset_factor": 1}
    scopes = {x: x_scope, y: y_scope, z: z_scope}
    binds = {
        t: tvm.tir.decl_buffer(
            t.shape, t.dtype, t.op.name, scope=scopes[t], **default_buffer_params
        )
        for t in [x, y, z]
    }

    return te.decl_tensor_intrin(
        z.op, _intrin_func, binds=binds, default_buffer_params=default_buffer_params
    )
