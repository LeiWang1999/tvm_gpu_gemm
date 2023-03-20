BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "int8", "shape": [128, 32, 32, 1280]}, "output0": {"dtype": "int8", "shape": [8, 32, 32, 80, 16, 16]}})' antares save ./direct_input_prmt_hwnc_int8.cu

BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "float16", "shape": [128, 32, 32, 1280]}, "output0": {"dtype": "float16", "shape": [8, 32, 32, 80, 16, 16]}})' antares save ./direct_input_prmt_hwnc_float16.cu

