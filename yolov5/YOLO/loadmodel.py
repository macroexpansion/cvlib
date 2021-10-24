import numpy as np
import torch
import torch.nn as nn

from .models.yolo import Detect, Model


class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU()
            if act is True
            else (
                act if isinstance(act, nn.Module) else nn.Identity()
            )
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(
        self, x, augment=False, profile=False, visualize=False
    ):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None, inplace=True, fuse=True):

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        ckpt = torch.load(w, map_location=map_location)
        if fuse:
            model.append(
                ckpt["ema" if ckpt.get("ema") else "model"]
                .float()
                .fuse()
                .eval()
            )  # FP32 model
        else:
            model.append(
                ckpt["ema" if ckpt.get("ema") else "model"]
                .float()
                .eval()
            )  # without layer fuse

    for m in model.modules():
        if type(m) in [
            nn.Hardswish,
            nn.LeakyReLU,
            nn.ReLU,
            nn.ReLU6,
            nn.SiLU,
            Detect,
            Model,
        ]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = (
                set()
            )  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        for k in ["names"]:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[
            torch.argmax(
                torch.tensor([m.stride.max() for m in model])
            ).int()
        ].stride  # max stride
        return model  # return ensemble