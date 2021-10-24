import cv2
import numpy as np

def test():
    print('test motion detector')

class MotionDetectorService:
    def __init__(self, set_para=False):
        self.first_frame = None
        self.count_frame = 0
        self.set_para = set_para
        if self.set_para == False:
            self.MAX_FRAME_CHANGE = 20
            self.MIN_BOX = 5000
            self.MAX_BOX = 50000
            self.KSIZE_BLUR = 21
            self.MIN_THRESHOLD = 25
            self.MAX_THRESHOLD = 255
            self.DILATED_ITERATIONS = 25
            self.ERODED_ITERATIONS = 10
            self.IMAGE_WITH = 1280
            self.IMAGE_HEIGHT = 720

    def set_parameter(
        self,
        maxframe_change=20,
        min_box=5000,
        max_box=50000,
        ksize_blur=21,
        min_threshold=25,
        max_threshold=255,
        dilated_iterations=25,
        eroded_iterations=10,
        image_with=1280,
        image_heith=720,
    ):
        if self.set_para == True:
            self.MAX_FRAME_CHANGE = maxframe_change
            self.MIN_BOX = min_box
            self.MAX_BOX = max_box
            self.KSIZE_BLUR = ksize_blur
            self.MIN_THRESHOLD = min_threshold
            self.MAX_THRESHOLD = max_threshold
            self.DILATED_ITERATIONS = dilated_iterations
            self.ERODED_ITERATIONS = eroded_iterations
            self.IMAGE_WITH = image_with
            self.IMAGE_HEIGHT = image_heith

    def predict(self, image):
        transfrom_image = self.transform(image)
        accepted_contours = []

        if (self.first_frame is None) or (
            self.count_frame > self.MAX_FRAME_CHANGE
        ):
            self.first_frame = np.float32(transfrom_image)
            self.count_frame = 0
            return accepted_contours

        # Running average on blur image
        ra_image = self.runningAverage(transfrom_image)
        subtract_image = cv2.absdiff(transfrom_image, ra_image)
        contours = self.processContours(subtract_image)
        self.count_frame += 1

        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)

            if (cnt_area < self.MIN_BOX) or (cnt_area > self.MAX_BOX):
                continue

            cnt_boxes = cv2.boundingRect(cnt)  # x,y,w,h
            accepted_contours.append(cnt_boxes)

        return accepted_contours

    def transform(self, image):
        image_copy = image.copy()

        image_size = [self.IMAGE_WITH, self.IMAGE_HEIGHT]

        image_copy = cv2.resize(image_copy, tuple(image_size))
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

        if type(self.KSIZE_BLUR) == list:
            k_size_blur = self.KSIZE_BLUR
        else:
            k_size_blur = [self.KSIZE_BLUR, self.KSIZE_BLUR]

        blur = cv2.GaussianBlur(gray, tuple(k_size_blur), 0)
        return blur

    def runningAverage(self, image):
        """[Update running average by calculates
            the weighted sum of img and accumulator dst
            if mask(x, y) # 0 => dst = (1 - alpha) * dst(x, y) + alpha * src(x, y)
        ]

        Args:
            image ([numpy array]): [Input image]

        Returns:
            [numpy array]: [Running average image]
        """

        cv2.accumulateWeighted(image, self.first_frame, 0.03)

        # Converting the matrix elements to
        # absolute values and converting the result to 8-bit
        return cv2.convertScaleAbs(self.first_frame)

    def processContours(self, subtract_image):
        """[Get contours from image after subtract]

        Args:
            subtract_image ([numpy array]): [Subtract path from 2 image]

        Returns:
            [list]: [Contours list]
        """
        thresh = cv2.threshold(
            subtract_image,
            self.MIN_THRESHOLD,
            self.MAX_THRESHOLD,
            cv2.THRESH_BINARY,
        )[1]

        # remove small object
        dilated_image = cv2.dilate(
            thresh, None, iterations=self.DILATED_ITERATIONS
        )

        # increase true object
        eroded_image = cv2.dilate(
            dilated_image, None, iterations=self.ERODED_ITERATIONS
        )

        contours, hierarchy = cv2.findContours(
            eroded_image.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return contours
