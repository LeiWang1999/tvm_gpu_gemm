# DEVICE_ID=3 BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[M / 16, N / 16, M % 16, N % 16] =. input0[M, N]", input_dict={"input0": {"dtype": "float16", "shape": [16384, 16384]}, "output0": {"dtype": "float16", "shape": [1024, 1024, 16, 16]}})' antares save ./a_permutation_16384.cu

# DEVICE_ID=3 BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[M / 16, N / 16, M % 16, N % 16] =. input0[M, N]", input_dict={"input0": {"dtype": "float16", "shape": [8192, 8192]}, "output0": {"dtype": "float16", "shape": [512, 512, 16, 16]}})' antares save ./a_permutation_8192.cu

DEVICE_ID=3 BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[M // 16, N // 8, M % 16, N % 8] =. input0[M, N]", input_dict={"input0": {"dtype": "float32", "shape": [16384, 16384]}, "output0": {"dtype": "float32", "shape": [1024, 2048, 16, 8]}})' antares save ./a_permutation_16384_float32.cu

# DEVICE_ID=3 BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[MM, NN, M, N] = input0[MM * 16 + M, NN * 8 + N] where MM in 1024, NN in 2048, M in 16, N in 8", input_dict={"input0": {"dtype": "float32", "shape": [16384, 16384]}, "output0": {"dtype": "float32", "shape": [1024, 2048, 16, 8]}})' antares save ./a_permutation_16384_float32_atomic.cu
