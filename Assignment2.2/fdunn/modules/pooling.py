"""
Pooling

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/pooling.py
"""
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from fdunn.modules.base import Module


class MaxPool2d(Module):
    """Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
    """

    def __init__(self, kernel_size, stride):
        # input and output
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.params = None
        self.grads = None

    def forward(self, input: np.ndarray):
        self.input = input
        ###########################################################################
        # TOaDO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if input.ndim == 3:
            input = input.reshape(1, *input.shape)
        assert (
            input.ndim == 4
        ), "Only 3D and 4D inputs are supported, got{}D instead".format(input.ndim)

        def funit(xi: np.ndarray, kernel_size: tuple, stride: tuple) -> np.ndarray:
            *a, hx, wx = xi.shape
            hk, wk = kernel_size
            hs, ws = stride
            hy = int(np.floor((hx - hk) / hs)) + 1
            wy = int(np.floor((wx - wk) / ws)) + 1
            yi = np.zeros((a + [hy, wy]))
            for i in range(hy):
                for j in range(wy):
                    temp = xi[:, :, i * hs : i * hs + hk, j * ws : j * ws + wk]
                    yi[:, :, i, j] = np.max(temp, axis=(2, 3))
            return yi

        if self.input.ndim == 3:
            output = funit(input, self.kernel_size, self.stride)[0]
        else:
            output = funit(input, self.kernel_size, self.stride)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad: np.ndarray):
        ###########################################################################
        # TaODO:                                                                   #
        # Implement the backward method.                                          #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if output_grad.ndim == 3:
            output_grad = output_grad.reshape(1, *output_grad.shape)
        assert (
            output_grad.ndim == 4
        ), "Only 3D and 4D outputgrad are supported, got{}D instead".format(
            output_grad.ndim
        )
        if self.input.ndim == 3:
            input = self.input.reshape(1, *self.input.shape)
        else:
            input = self.input
        input_grad = np.zeros(input.shape)

        for i in range(output_grad.shape[2]):
            for j in range(output_grad.shape[3]):
                h0 = i * self.stride[0]
                h1 = h0 + self.kernel_size[0]
                w0 = j * self.stride[1]
                w1 = w0 + self.kernel_size[1]
                slice1 = input[:, :, h0:h1, w0:w1]
                maxval = np.max(slice1, axis=(2, 3), keepdims=True)
                mask = slice1 == maxval
                input_grad[:, :, h0:h1, w0:w1] += (
                    output_grad[:, :, i, j][:, :, np.newaxis, np.newaxis] * mask
                )
        if self.input.ndim == 3:
            input_grad = input_grad[0]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return input_grad


if __name__ == "__main__":
    a = np.random.rand(3, 5, 8)
    m = MaxPool2d((2, 3), 2)
    b = m.forward(a)
    print(b.shape)
    outgrad = np.random.rand(*b.shape)
    c = m.backward(outgrad)
    print(c.shape)
