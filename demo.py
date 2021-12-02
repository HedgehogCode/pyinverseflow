import os
import numpy as np
import imageio
import matplotlib.pyplot as plt

from pyinverseflow import inverse_flow


def read_flo(filename):
    """Read a .flo file.
    See https://vision.middlebury.edu/flow/code/flow-code/flowIO.cpp
    """
    with open(filename, "rb") as f:
        format = f.read(4)
        if format != b"PIEH":
            raise ValueError("Wrong file format")
        w = int.from_bytes(f.read(4), "little")
        h = int.from_bytes(f.read(4), "little")
        data = np.frombuffer(f.read(), "<f")
        reshaped = data.reshape((h, w, 2))[..., ::-1]
        return np.ascontiguousarray(reshaped)


# Read inputs
img1 = imageio.imread(os.path.join("inverse_flow", "I1.png")).astype(np.float32) / 255.0
img2 = imageio.imread(os.path.join("inverse_flow", "I2.png")).astype(np.float32) / 255.0
forward_flow = read_flo(os.path.join("inverse_flow", "ground_truth.flo"))

# Compute the backwards flow
strategy = "max_flow"  # Try "max_flow", "avg_flow", "max_image", "avg_image"
fill = "oriented"  # Try "min", "avg", "oriented", "none"
backward_flow, mask = inverse_flow(
    forward_flow, img1=img1, img2=img2, strategy=strategy, fill=fill
)

# Draw the flow and mask
fig, ax = plt.subplots(1, 3, squeeze=False)
# u
ax[0, 0].set_title("u")
ax[0, 0].imshow(backward_flow[..., 1])
# v
ax[0, 1].set_title("v")
ax[0, 1].imshow(backward_flow[..., 0])
# mask
ax[0, 2].set_title("mask")
ax[0, 2].imshow(mask)

plt.show()
