import torch
import torch.nn.functional as F


class GLELinear(torch.nn.Linear):

    def compute_error(self, e):
        return torch.mm(e, self.weight)

    def compute_grad(self, r_bottom, e):
        self.weight.grad = - torch.bmm(e.unsqueeze(2), r_bottom.unsqueeze(1)).mean(0)
        if self.bias is not None:
            self.bias.grad = - e.mean(0)


class GLEConv(torch.nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_error(self, e, bottom_r_shape=None):
        # Calculate the expected output shape of the transpose convolution
        # This is the formula for ConvTranspose2d output size
        output_height = (e.shape[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        output_width = (e.shape[3] - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]

        # Calculate the necessary output_padding
        output_padding_h = bottom_r_shape[2] - output_height if bottom_r_shape else 0
        output_padding_w = bottom_r_shape[3] - output_width if bottom_r_shape else 0
        output_padding = (output_padding_h, output_padding_w)

        return F.conv_transpose2d(e, self.weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def compute_grad(self, r_bottom, e):
        grad_weight = torch.nn.grad.conv2d_weight(r_bottom, self.weight.shape, e, self.stride, self.padding)
        self.weight.grad = - grad_weight
        if self.bias is not None:
            self.bias.grad = - e.mean((0, 2, 3))
