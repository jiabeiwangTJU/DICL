import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS
from mmcv.ops import DeformConv2dPack

@NECKS.register_module()
class FPNs16C45add(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_dconv=False,
                 kernel1=True):
        super(FPNs16C45add, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_dconv = use_dconv
        self.kernel1 = kernel1
        self.deform_conv = DeformConv2dPack(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1, )
        if self.use_dconv:
            if self.kernel1:
                self.lateral_conv_8 = DeformConv2dPack(
                    in_channels[0],
                    out_channels,
                    kernel_size=1,)

                self.lateral_conv_32 = DeformConv2dPack(
                    in_channels[2],
                    out_channels,
                    kernel_size=1,)
            else:
                self.lateral_conv_8 = DeformConv2dPack(
                    in_channels[0],
                    out_channels,
                    kernel_size=3,
                    padding=1,)

                self.lateral_conv_32 = DeformConv2dPack(
                    in_channels[2],
                    out_channels,
                    kernel_size=3,
                    padding=1,)
        else:
            self.lateral_conv_8 = nn.Conv2d(
                    in_channels[0],
                    out_channels,
                    kernel_size=1,)

            self.lateral_conv_32 = nn.Conv2d(
                    in_channels[2],
                    out_channels,
                    kernel_size=1,)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # print(len(inputs), inputs[0].shape, inputs[1].shape, inputs[2].shape),

        new_input_32 = self.lateral_conv_32(inputs[2])

        res_x = inputs[1] + new_input_32 #
        x = self.deform_conv(res_x)
        x = x + res_x

        return tuple([x])
