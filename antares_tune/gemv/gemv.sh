# DEVICE_ID=3 BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K].cast(`int32`) * input1[K, M].cast(`int32`)", { "input0": {"dtype": "int8", "shape": [18966528, 25]}, "input1": {"dtype": "int8", "shape": [1, 25]}})' antares save ./gemv_s8_s32.cu

DEVICE_ID=0 BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K].cast(`int32`) * input1[K, M].cast(`int32`)", { "input0": {"dtype": "int8", "shape": [18966528, 32]}, "input1": {"dtype": "int8", "shape": [1, 32]}})' antares save ./gemv_s8_s32_align.cu
