import numpy as np
# The sizes of inputs and filters
batch_size = 128
height = 42
width = 42
in_channels = 1024
out_channels = 384
kernel_h = 1
kernel_w = 1
pad_h = 0
pad_w = 0
stride_h = 1
stride_w = 1
dilation_h = 1
dilation_w = 1
output_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
output_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
print("output_height: ", output_height)
print("output_width: ", output_width)

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size,
    height,
    width,
    in_channels
)

# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    out_channels,
    kernel_h,
    kernel_w,
    in_channels,
)

# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size,
    output_height,
    output_width,
    out_channels,
)

# compute conv2d with layout nhwc nhwc nhwc
# Input feature map: (N, H, W, IC, n, ic)
data =  np.mod(np.arange(np.prod(data_shape)), 10).reshape(data_shape).astype("float16")
# Kernel: (H, W, IC, OC, ic, oc)
kernel = np.mod(np.arange(np.prod(kernel_shape)), 10).reshape(kernel_shape).astype("float16")
# Output feature map: (N, H, W, OC, n, oc)
output = np.zeros(output_shape).astype(np.float16)

import torch
torch.backends.cudnn.allow_tf32 = True
# convert a from nhwc to nchw
a_np = np.transpose(data, (0, 3, 1, 2))
a_torch = torch.tensor(a_np, device="cuda")
    a_torch = torch.nn.functional.pad(a_torch, (pad_h, pad_h, pad_w, pad_w))
# convert b from ohwi to oihw
b_np = np.transpose(kernel, (0, 3, 1, 2))
b_torch = torch.tensor(b_np, device="cuda")
c_torch = torch.nn.functional.conv2d(a_torch, b_torch)
# convert c from nchw to nhwc
c_torch_np = np.transpose(c_torch.cpu().numpy(), (0, 2, 3, 1))

print("torch result: ", c_torch_np[0][0][0][0:10])
