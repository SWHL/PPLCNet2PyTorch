# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import six
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

from utils import (CropImage, DecodeImage, NormalizeImage, ResizeImage,
                   ToCHWImage)

MODEL_URLS = {
    "PPLCNet_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
    "PPLCNet_x0_35":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams",
    "PPLCNet_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams",
    "PPLCNet_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
    "PPLCNet_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams",
    "PPLCNet_x1_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams",
    "PPLCNet_x2_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams",
    "PPLCNet_x2_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

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


def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    return


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super().__init__()

        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = BatchNorm(
            num_filters,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        # x.shape: 1x16x112x112
        x = self.conv(x)
        # x.shape 1x32x112x112
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)

        if self.use_se:
            x = self.se(x)

        x = self.pw_conv(x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class PPLCNet(nn.Layer):
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
            stride=2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
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

        self.avg_pool = AdaptiveAvgPool2D(1)

        self.last_conv = Conv2D(
            in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)

        self.hardswish = nn.Hardswish()
        self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)

        self.fc = Linear(self.class_expand, class_num)

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


def _load_pretrained(pretrained, model, *args):
    if isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def PPLCNet_x0_25(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_25` model depends on args.
    """
    model = PPLCNet(scale=0.25, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_25"], use_ssld)
    return model


def PPLCNet_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_35` model depends on args.
    """
    model = PPLCNet(scale=0.35, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_35"], use_ssld)
    return model


def PPLCNet_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_5` model depends on args.
    """
    model = PPLCNet(scale=0.5, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_5"], use_ssld)
    return model


def PPLCNet_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_75
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_75` model depends on args.
    """
    model = PPLCNet(scale=0.75, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_75"], use_ssld)
    return model


def PPLCNet_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_0` model depends on args.
    """
    model = PPLCNet(scale=1.0, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x1_0"], use_ssld)
    return model


def PPLCNet_x1_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x1_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_5` model depends on args.
    """
    model = PPLCNet(scale=1.5, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x1_5"], use_ssld)
    return model


def PPLCNet_x2_0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x2_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x2_0` model depends on args.
    """
    model = PPLCNet(scale=2.0, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x2_0"], use_ssld)
    return model


def PPLCNet_x2_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x2_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x2_5` model depends on args.
    """
    model = PPLCNet(scale=2.5, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x2_5"], use_ssld)
    return model



if __name__ == '__main__':
    decode_img = DecodeImage(to_rgb=True, channel_first=False)
    resize_img = ResizeImage(resize_short=256)
    crop_img = CropImage(size=224)

    scale = 1 / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize_img = NormalizeImage(scale, mean, std, order='')

    to_chw_img = ToCHWImage()

    img_path = 'images/n01440764_9780.JPEG'
    with open(img_path, 'rb') as f:
        x = f.read()

    x = decode_img(x)
    x = resize_img(x)
    x = crop_img(x)
    x = normalize_img(x)
    x = to_chw_img(x)

    batch_data = []
    batch_data.append(x)
    batch_tensor = paddle.to_tensor(batch_data)

    with open('images/imagenet1k_label_list.txt', 'r', encoding='utf-8') as f:
        label_list = f.readlines()

    model_path = 'pretrained_models/PPLCNet_x1_0_pretrained'
    model = PPLCNet_x1_0(pretrained=model_path)
    model.eval()

    y = model(batch_tensor)
    y = F.softmax(y, axis=-1)

    y = y.numpy()
    probs = y[0]
    index = probs.argsort(axis=0)[-1:][::-1][0]
    score = probs[index]
    print(f'{label_list[index].strip()}: {score}')
