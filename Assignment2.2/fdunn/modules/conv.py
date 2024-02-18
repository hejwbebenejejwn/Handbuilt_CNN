"""
Conv2D

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
"""
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from fdunn.modules.base import Module


class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        # input and output
        self.input = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # params
        self.params = {}
        ###########################################################################
        # TOaDO:                                                                   #
        # Implement the params init.                                              #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.padding = (
            self.padding
            if isinstance(self.padding, tuple)
            else (self.padding, self.padding)
        )
        self.stride = (
            self.stride
            if isinstance(self.stride, tuple)
            else (self.stride, self.stride)
        )
        self.kernel_size = (
            self.kernel_size
            if isinstance(self.kernel_size, tuple)
            else (self.kernel_size, self.kernel_size)
        )
        k = 1 / in_channels / self.kernel_size[0] / self.kernel_size[1]
        if bias:
            self.params["b"] = np.random.uniform(
                -np.sqrt(k), np.sqrt(k), (out_channels,)
            )
        self.params["W"] = np.random.uniform(
            -np.sqrt(k),
            np.sqrt(k),
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
        )
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # grads of params
        self.grads = {}

    def forward(self, input):
        self.input = input

        ###########################################################################
        # TODaO:                                                                   #
        # Implement the forward method.                                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if input.ndim == 3:
            input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2])
        assert (
            input.ndim == 4
        ), "Only 3D and 4D inputs are supported, got {}D instead".format(input.ndim)

        def padding(x: np.ndarray, pad: tuple):
            return np.pad(
                x, pad_width=((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1]))
            )

        def unit(xi: np.ndarray, wj: np.ndarray, b: int, stride: tuple):
            """xi:(Cin,Hin,Win),
            wj:(Cin,KS[0],KS[1]),
            """
            hi, wi = xi.shape[1:]
            k0, k1 = wj.shape[1:]
            ho = int(np.floor((hi - k0) / stride[0]) + 1)
            wo = int(np.floor((wi - k1) / stride[1]) + 1)
            y = np.zeros((ho, wo))
            for i in range(ho):
                for j in range(wo):
                    temp = xi[
                        :,
                        i * stride[0] : i * stride[0] + k0,
                        j * stride[1] : j * stride[1] + k1,
                    ]
                    y[i, j] = np.sum(temp * wj)
            return y + b

        self.input_padded = padding(input, self.padding)

        output = np.stack(
            [
                np.stack(
                    [
                        unit(
                            self.input_padded[i],
                            self.params["W"][j],
                            self.params["b"][j],
                            self.stride,
                        )
                        for j in range(self.params["W"].shape[0])
                    ],
                    axis=0,
                )
                for i in range(self.input_padded.shape[0])
            ],
            axis=0,
        )
        if self.input.ndim == 3:
            output = output[0]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad):
        ###########################################################################
        # TOaDO:                                                                   #
        # Implement the backward method.                                          #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if output_grad.ndim == 3:
            output_grad = output_grad.reshape(1, *output_grad.shape)
        assert (
            output_grad.ndim == 4
        ), "Only 3D and 4D output gradients are supported, got {}D instead".format(
            output_grad.ndim
        )

        def wgrad_ij(output_grad: np.ndarray, input: np.ndarray, i: int, j: int):
            inputij = input[
                :,
                :,
                i : i
                + self.stride[0] * (output_grad.shape[2] - 1)
                + 1 : self.stride[0],
                j : j
                + self.stride[1] * (output_grad.shape[3] - 1)
                + 1 : self.stride[1],
            ]

            output_grad_flatten = np.stack(
                [
                    output_grad[:, p, :, :].flatten()
                    for p in range(output_grad.shape[1])
                ],
                axis=0,
            )
            inputij_flatten = np.stack(
                [inputij[:, p, :, :].flatten() for p in range(inputij.shape[1])], axis=0
            )
            return output_grad_flatten.dot(inputij_flatten.T)

        def xgradcd(output_grad: np.ndarray, c: int, d: int):
            a0 = int(
                np.ceil(
                    (self.padding[0] + c + 1 - self.kernel_size[0]) / self.stride[0]
                )
            )
            a1 = int(np.floor((self.padding[0] + c) / self.stride[0])) + 1
            b0 = int(
                np.ceil(
                    (self.padding[1] + d + 1 - self.kernel_size[1]) / self.stride[1]
                )
            )
            b1 = int(np.floor((self.padding[1] + d) / self.stride[1])) + 1

            output_grad_selected = output_grad[:, :, a0:a1, b0:b1]

            Wcd = self.params["W"][
                :,
                :,
                c + self.padding[0] - a0 * self.stride[0] : None
                if c + self.padding[0] - a1 * self.stride[0] < 0
                else c + self.padding[0] - a1 * self.stride[0] : -self.stride[0],
                d + self.padding[1] - b0 * self.stride[1] : None
                if d + self.padding[1] - b1 * self.stride[1] < 0
                else d + self.padding[1] - b1 * self.stride[1] : -self.stride[1],
            ]
            output_grad_flatten = np.stack(
                [
                    output_grad_selected[i, :, :, :].flatten()
                    for i in range(output_grad_selected.shape[0])
                ],
                axis=0,
            )
            Wcd_flatten = np.stack(
                [Wcd[:, i, :, :].flatten() for i in range(Wcd.shape[1])], axis=0
            )
            return output_grad_flatten.dot(Wcd_flatten.T)

        self.grads["b"] = output_grad.sum(axis=0).sum(axis=1).sum(axis=1)
        self.grads["W"] = np.stack(
            [
                np.stack(
                    [
                        wgrad_ij(
                            output_grad=output_grad, input=self.input_padded, i=i, j=j
                        )
                        for i in range(self.params["W"].shape[2])
                    ],
                    axis=1,
                )
                for j in range(self.params["W"].shape[3])
            ],
            axis=1,
        ).swapaxes(1, 3)

        input_grad = np.stack(
            [
                np.stack(
                    [
                        xgradcd(output_grad=output_grad, c=c, d=d)
                        for c in range(self.input.shape[-2])
                    ],
                    axis=1,
                )
                for d in range(self.input.shape[-1])
            ],
            axis=1,
        ).swapaxes(1, 3)
        if self.input.ndim == 3:
            input_grad = input_grad[0]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return input_grad


if __name__ == "__main__":
    lc = Conv2d(2, 3, 4, stride=(2, 3), padding=(3, 2))
    x = np.random.rand(2, 4, 4)
    out = lc.forward(x)
    outgrad = np.random.rand(*out.shape)
    lc.backward(outgrad)
