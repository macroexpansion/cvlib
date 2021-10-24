from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from .models.common import Conv
from .utils.augmentations import letterbox
from .utils.general import (check_img_size, is_ascii, non_max_suppression,
                            scale_coords)
from .utils.torch_utils import load_classifier


class YOLOv5(object):
    @torch.no_grad()
    def __init__(
        self, weights, modelyolo, use_cuda=True, set_para=False
    ):
        self.weights = weights
        self.set_para = set_para
        self.use_cuda = use_cuda
        if self.use_cuda == True:
            self.cuda = True
        else:
            self.cuda = False

        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.half = self.device.type != "cpu"

        self.model = modelyolo
        self.stride = int(self.model.stride.max())
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        if self.half:
            self.model.half()  # to FP16

        imgsz = 640
        self.imgsz = check_img_size(
            imgsz, s=self.stride
        )  # check image size
        ascii = is_ascii(self.names)
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, *imgsz)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )
        if self.set_para == False:
            self.conf_thres = 0.25
            self.iou_thres = 0.45
            self.max_det = 1000

    def set_parameter(self, conf_thres, iou_thres, max_det):
        if self.set_para == True:
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.max_det = max_det

    def detector(self, image):
        img = letterbox(image, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = (
            img.half() if self.half else img.float()
        )  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]

        pred = self.model(img, augment=False, visualize=False)[0]
        # NMS
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            None,
            False,
            max_det=self.max_det,
        )

        bounding_boxs, classId, labels, scores = [], [], [], []
        for i, det in enumerate(pred):
            im0 = image.copy()
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape
                ).round()
                for *xyxy, conf, cls in reversed(det):
                    classId.append(int(cls))
                    labels.append(self.names[int(cls)])
                    scores.append(round(float(conf), 3))
                    bounding_boxs.append(
                        [
                            int(xyxy[0]),
                            int(xyxy[1]),
                            int(xyxy[2]),
                            int(xyxy[3]),
                        ]
                    )

        return bounding_boxs, classId, labels, scores
