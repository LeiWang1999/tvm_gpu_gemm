# nhwc * ohwi(nhwc)s

BACKEND=c-cuda STEP=1000 COMPUTE_V1='- einstein_v2("output0[N, H, W, C] = input0[N, H, W, C].when([N < 128, H < 42, W < 42, C < 1008], const(0.0).cast(`float16`)) where N in 128, H in 42, W in 42, C in 1024 ", input_dict={"input0": {"dtype": "float16", "shape": [128, 42, 42, 1008]}})' antares save ./padding_from128_42_42_1008_to_128_42_42_1024.cu
