from __future__ import print_function

import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .data import cfg_mnet, cfg_re50
from .layers.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class RetinaFaceDetector(object):
    def __init__(
        self, network, path_model, path_checkpoint=None, use_cuda=True
    ):

        self.network = network
        self.path_model = path_model
        self.path_checkpoint = path_checkpoint
        self.use_cuda = use_cuda
        if self.use_cuda == True:
            self.cpu = False
        else:
            self.cpu = True

    def load_model(self):

        torch.set_grad_enabled(False)
        if self.network == "resnet50":
            self.cfg = cfg_re50
            self.net = RetinaFace(cfg=self.cfg)
        elif self.network == "mobile0.25":
            self.cfg = cfg_mnet
            self.net = RetinaFace(
                cfg=self.cfg, path_checkpoint=self.path_checkpoint
            )

        if self.cpu:
            pretrained_dict = torch.load(
                self.path_model,
                map_location=lambda storage, loc: storage,
            )
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                self.path_model,
                map_location=lambda storage, loc: storage.cuda(
                    device
                ),
            )

        rm_prefix = (
            lambda x: x.split("module.", 1)[-1]
            if x.startswith("module.")
            else x
        )
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = {
                rm_prefix(key): value
                for key, value in pretrained_dict[
                    "state_dict"
                ].items()
            }
        else:
            pretrained_dict = {
                rm_prefix(key): value
                for key, value in pretrained_dict.items()
            }
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()

    def set_parameter(
        self,
        width,
        height,
        threshold=0.8,
        nms_threshold=0.4,
        top_k=50,
        keep_top_k=20,
    ):
        self.confidence_threshold = threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k

        self.im_width = width
        self.im_height = height
        self.device = torch.device("cpu" if self.cpu else "cuda")

        self.scale = torch.Tensor(
            [
                self.im_width,
                self.im_height,
                self.im_width,
                self.im_height,
            ]
        )
        self.scale = self.scale.to(self.device)

        priorbox = PriorBox(
            self.cfg, image_size=(self.im_height, self.im_width)
        )
        priors = priorbox.forward()
        priors = priors.to(self.device)
        self.prior_data = priors.data

    def detect_face(self, img_raw):
        resize = 1
        cudnn.benchmark = True

        net = self.net.to(self.device)

        img_input = cv2.resize(
            img_raw, (self.im_width, self.im_height)
        )
        img = img_input
        img = np.float32(img_input)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        loc, conf, landms = net(img)

        boxes = decode(
            loc.data.squeeze(0), self.prior_data, self.cfg["variance"]
        )
        boxes = boxes * self.scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(
            landms.data.squeeze(0),
            self.prior_data,
            self.cfg["variance"],
        )

        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[: self.keep_top_k, :]
        landms = landms[: self.keep_top_k, :]
        return dets, landms
