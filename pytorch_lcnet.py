# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import torch
import torch.nn as nn

NET_CONFIG = {
    #           k, in_c, out_c, s, use_se
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self, num_channels, filter_size,
                 num_filters, stride, num_groups=1):
        super().__init__()

        # TODO: 显示指定权重初始化方式
        self.conv = nn.Conv2d(in_channels=num_channels,
                              out_channels=num_filters,
                              kernel_size=filter_size,
                              stride=stride,
                              padding=(filter_size - 1) // 2,
                              groups=num_groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(num_features=num_filters)

        self.hard_wish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hard_wish(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel,
                               out_channels=channel // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=channel // reduction,
                               out_channels=channel,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.hard_sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.hard_sigmoid(x)
        x = torch.mul(identity, x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self, num_channels, num_filters,
                 stride, dw_size=3, use_se=False):
        super().__init__()
        self.use_se = use_se

        self.dw_conv = ConvBNLayer(num_channels=num_channels,
                                   num_filters=num_filters,
                                   filter_size=dw_size,
                                   stride=stride,
                                   num_groups=num_channels)

        if use_se:
            self.se = SEModule(num_channels)

        self.pw_conv = ConvBNLayer(num_channels=num_channels,
                                   filter_size=1,
                                   num_filters=num_filters,
                                   stride=1)

    def forward(self, x):
        x = self.dw_conv(x)

        if self.use_se:
            x = self.se(x)

        x = self.pw_conv(x)
        return x


class PyTorchLCNet(nn.Module):
    def __init__(self,
                 scale=1.0,
                 class_num=1000,
                 dropout_prob=0.2,
                 class_expand=1280):
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2
        )

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for k, in_c, out_c, s, se in NET_CONFIG["blocks2"]
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(
            in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.hard_swish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.flatten = nn.Flatten(start_dim=1, end_dim=1)

        self.fc = nn.Linear(self.class_expand, class_num)

    def forward(self, x):
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def PyTorchLCNet_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    PyTorchLCNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_0` model depends on args.
    """
    model = PyTorchLCNet(scale=1.0, **kwargs)
    # _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x1_0"], use_ssld)
    return model


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = PyTorchLCNet_x1_0()

    y = model(x)
    print(y.shape)
