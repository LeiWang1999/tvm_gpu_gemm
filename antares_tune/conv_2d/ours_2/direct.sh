BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "int8", "shape": [128, 56, 56, 128]}, "output0": {"dtype": "int8", "shape": [56, 56, 8, 32, 16, 16]}})' antares save ./direct_input_prmt_hwnc_int8.cu

BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "float16", "shape": [128, 56, 56, 128]}, "output0": {"dtype": "float16", "shape": [56, 56, 8, 32, 16, 16]}})' antares save ./direct_input_prmt_hwnc_float16.cu


# BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "int8", "shape": [512, 3, 3, 256]}, "output0": {"dtype": "int8", "shape": [3, 3, 32, 16, 16, 16]}})' antares save ./direcit_weight_prmt_int8.cu


