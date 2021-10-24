import time

import cv2
import numpy as np
import torch
from torch.autograd.variable import Variable

from .mtcnn import image_tools, utils
from .mtcnn.models import ONet, PNet, RNet


class MtcnnDetector(object):
    """
    P,R,O net face detection and landmarks align
    """

    def __init__(
        self, path_pmodel, path_rmodel, path_omodel, use_cuda=True
    ):
        self.path_pnet = path_pmodel
        self.path_rnet = path_rmodel
        self.path_onet = path_omodel
        self.use_cuda = use_cuda

    def load_model(self):
        self.pnet_detector = PNet(use_cuda=self.use_cuda)
        if self.use_cuda:
            self.pnet_detector.load_state_dict(
                torch.load(self.path_pnet)
            )
            self.pnet_detector.cuda()
        else:
            self.pnet_detector.load_state_dict(
                torch.load(
                    self.path_pnet,
                    map_location=lambda storage, loc: storage,
                )
            )
        self.pnet_detector.eval()

        self.rnet_detector = RNet(use_cuda=self.use_cuda)
        if self.use_cuda:
            self.rnet_detector.load_state_dict(
                torch.load(self.path_rnet)
            )
            self.rnet_detector.cuda()
        else:
            self.rnet_detector.load_state_dict(
                torch.load(
                    self.path_rnet,
                    map_location=lambda storage, loc: storage,
                )
            )
        self.rnet_detector.eval()

        self.onet_detector = ONet(use_cuda=self.use_cuda)
        if self.use_cuda:
            self.onet_detector.load_state_dict(
                torch.load(self.path_onet)
            )
            self.onet_detector.cuda()
        else:
            self.onet_detector.load_state_dict(
                torch.load(
                    self.path_onet,
                    map_location=lambda storage, loc: storage,
                )
            )
        self.onet_detector.eval()

    def set_parameter(
        self,
        threshold=[0.6, 0.7, 0.7],
        scale_factor=0.709,
        min_face_size=20,
        stride=2,
    ):
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.stride = stride
        self.thresh = threshold

    def square_bbox(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x m
                input bbox
        Returns:
        -------
            a square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        l = np.maximum(h, w)

        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1
        return square_bbox

    def generate_bounding_box(self, map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            map: numpy array , n x m x 1
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        cellsize = 12  # receptive field
        t_index = np.where(map > threshold)
        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        # choose bounding box whose socre are larger than threshold
        dx1, dy1, dx2, dy2 = [
            reg[0, t_index[0], t_index[1], i] for i in range(4)
        ]
        reg = np.array([dx1, dy1, dx2, dy2])
        score = map[t_index[0], t_index[1], 0]
        boundingbox = np.vstack(
            [
                np.round(
                    (stride * t_index[1]) / scale
                ),  # x1 of prediction box in original image
                np.round(
                    (stride * t_index[0]) / scale
                ),  # y1 of prediction box in original image
                np.round(
                    (stride * t_index[1] + cellsize) / scale
                ),  # x2 of prediction box in original image
                np.round(
                    (stride * t_index[0] + cellsize) / scale
                ),  # y2 of prediction box in original image
                score,
                reg,
            ]
        )

        return boundingbox.T

    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel
                input image, channels in BGR order here
            scale: float number
                scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(
            img, new_dim, interpolation=cv2.INTER_LINEAR
        )  # resized image
        return img_resized

    def pad(self, bboxes, w, h):
        """
            pad the the boxes
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        # width and height
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
        numbox = bboxes.shape[0]

        dx = np.zeros((numbox,))
        dy = np.zeros((numbox,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1
        x, y, ex, ey = (
            bboxes[:, 0],
            bboxes[:, 1],
            bboxes[:, 2],
            bboxes[:, 3],
        )

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array
            one batch

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """

        # original wider face data
        h, w, c = im.shape
        net_size = 12
        current_scale = (
            float(net_size) / self.min_face_size
        )  # find initial scale
        im_resized = self.resize_image(
            im, current_scale
        )  # scale = 1.0
        current_height, current_width, _ = im_resized.shape

        # fcn
        all_boxes = list()
        i = 0
        while min(current_height, current_width) > net_size:
            # print(i)
            feed_imgs = []
            image_tensor = image_tools.convert_image_to_tensor(
                im_resized
            )
            feed_imgs.append(image_tensor)
            feed_imgs = torch.stack(feed_imgs)
            feed_imgs = Variable(feed_imgs)

            if self.pnet_detector.use_cuda:
                feed_imgs = feed_imgs.cuda()

            cls_map, reg = self.pnet_detector(feed_imgs)
            cls_map_np = image_tools.convert_chwTensor_to_hwcNumpy(
                cls_map.cpu()
            )
            reg_np = image_tools.convert_chwTensor_to_hwcNumpy(
                reg.cpu()
            )

            boxes = self.generate_bounding_box(
                cls_map_np[0, :, :],
                reg_np,
                current_scale,
                self.thresh[0],
            )

            # generate pyramid images
            current_scale *= (
                self.scale_factor
            )  # self.scale_factor = 0.709
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = utils.nms(boxes[:, :5], 0.5, "Union")
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = utils.nms(all_boxes[:, 0:5], 0.7, "Union")
        all_boxes = all_boxes[keep]

        # x2 - x1
        # y2 - y1
        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        boxes = np.vstack(
            [
                all_boxes[:, 0],
                all_boxes[:, 1],
                all_boxes[:, 2],
                all_boxes[:, 3],
                all_boxes[:, 4],
            ]
        )

        boxes = boxes.T
        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack(
            [
                align_topx,
                align_topy,
                align_bottomx,
                align_bottomy,
                all_boxes[:, 4],
            ]
        )
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        # im: an input image
        h, w, c = im.shape

        if dets is None:
            return None, None

        # return square boxes
        dets = self.square_bbox(dets)
        # rounds
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(
            dets, w, h
        )
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i] : edy[i] + 1, dx[i] : edx[i] + 1, :] = im[
                y[i] : ey[i] + 1, x[i] : ex[i] + 1, :
            ]
            crop_im = cv2.resize(tmp, (24, 24))
            crop_im_tensor = image_tools.convert_image_to_tensor(
                crop_im
            )
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg = self.rnet_detector(feed_imgs)
        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7)

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1
        boxes = np.vstack(
            [
                keep_boxes[:, 0],
                keep_boxes[:, 1],
                keep_boxes[:, 2],
                keep_boxes[:, 3],
                keep_cls[:, 0],
            ]
        )

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        boxes_align = np.vstack(
            [
                align_topx,
                align_topy,
                align_bottomx,
                align_bottomy,
                keep_cls[:, 0],
            ]
        )

        boxes = boxes.T
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes_align: numpy array
            boxes after calibration
        landmarks_align: numpy array
            landmarks after calibration

        """
        h, w, c = im.shape

        if dets is None:
            return None, None

        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(
            dets, w, h
        )
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            # crop input image
            tmp[dy[i] : edy[i] + 1, dx[i] : edx[i] + 1, :] = im[
                y[i] : ey[i] + 1, x[i] : ex[i] + 1, :
            ]
            crop_im = cv2.resize(tmp, (48, 48))
            crop_im_tensor = image_tools.convert_image_to_tensor(
                crop_im
            )
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]

        boxes_align = np.vstack(
            [
                align_topx,
                align_topy,
                align_bottomx,
                align_bottomy,
                keep_cls[:, 0],
            ]
        )

        boxes_align = boxes_align.T

        landmark = np.vstack(
            [
                align_landmark_topx + keep_landmark[:, 0] * bw,
                align_landmark_topy + keep_landmark[:, 1] * bh,
                align_landmark_topx + keep_landmark[:, 2] * bw,
                align_landmark_topy + keep_landmark[:, 3] * bh,
                align_landmark_topx + keep_landmark[:, 4] * bw,
                align_landmark_topy + keep_landmark[:, 5] * bh,
                align_landmark_topx + keep_landmark[:, 6] * bw,
                align_landmark_topy + keep_landmark[:, 7] * bh,
                align_landmark_topx + keep_landmark[:, 8] * bw,
                align_landmark_topy + keep_landmark[:, 9] * bh,
            ]
        )

        landmark_align = landmark.T

        return boxes_align, landmark_align

    def detect_face(self, img):
        """Detect face over image"""
        boxes_align = np.array([])
        landmark_align = np.array([])

        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])

        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(
                img, boxes_align
            )
            if boxes_align is None:
                return np.array([]), np.array([])

        return boxes_align, landmark_align
