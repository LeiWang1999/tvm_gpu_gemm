# import tvm convert
from tvm.runtime import convert
from tvm.tir.expr import Cast, IntImm
from tvm.tir.function import TensorIntrin
from tvm.script import tir as T
from tvm._ffi import register_func
lift = convert


def get_aync_copy_intrin(dtype, scope="shared"):
    if dtype == "float32":
        elems = 4
    elif dtype == "float16":
        elems = 8
    elif dtype == "int8":
        elems = 16
    else:
        raise ValueError("Unsupported dtype: {}".format(dtype))

    @T.prim_func
    def async_copy_desc(global_handle: T.handle, shared_handle: T.handle) -> None:
        globalVar = T.match_buffer(
            global_handle,
            (elems),
            dtype,
            align=64,
            offset_factor=elems,
            scope="global",
        )
        sharedVar = T.match_buffer(
            shared_handle, (elems), dtype, align=64, offset_factor=16, scope=scope
        )

        with T.block("root"):
            T.reads(globalVar[0:elems])
            T.writes(sharedVar[0:elems])

            for ax0 in T.vectorized(elems):
                with T.block("shared_warp"):
                    v0 = T.axis.remap("S", [ax0])
                    T.reads(globalVar[v0])
                    T.writes(sharedVar[v0])
                    sharedVar[v0] = globalVar[v0]

    @T.prim_func
    def async_copy_imlp(global_handle: T.handle, shared_handle: T.handle) -> None:
        globalVar = T.match_buffer(
            global_handle,
            (elems),
            dtype,
            align=64,
            offset_factor=elems,
            scope="global",
        )
        sharedVar = T.match_buffer(
            shared_handle, (elems), dtype, align=64, offset_factor=elems, scope=scope
        )

        with T.block("root"):
            T.reads(globalVar[0:elems])
            T.writes(sharedVar[0:elems])
            T.attr(0, "async_scope", 1)
            for ax0 in T.vectorized(elems):
                with T.block("shared_warp"):
                    v0 = T.axis.remap("S", [ax0])
                    T.reads(globalVar[v0])
                    T.writes(sharedVar[v0])
                    sharedVar[v0] = globalVar[v0]

    return async_copy_desc, async_copy_imlp


ASYNC_COPY_F16_X8_INTRIN = "async_copy.f16._x8"
TensorIntrin.register(ASYNC_COPY_F16_X8_INTRIN, *
                      get_aync_copy_intrin("float16"))

ASYNC_COPY_S8_X16_INTRIN = "async_copy.s8._x16"
TensorIntrin.register(ASYNC_COPY_S8_X16_INTRIN, *
                      get_aync_copy_intrin("int8"))

ASYNC_COPY_F16_X8_INTRIN_DYN = "async_copy.f16._x8_DYN"
TensorIntrin.register(ASYNC_COPY_F16_X8_INTRIN_DYN, *
                      get_aync_copy_intrin("float16", scope="shared.dyn"))

ASYNC_COPY_S8_X16_INTRIN_DYN = "async_copy.s8._x16_DYN"
TensorIntrin.register(ASYNC_COPY_S8_X16_INTRIN_DYN, *
                      get_aync_copy_intrin("int8", scope="shared.dyn"))
