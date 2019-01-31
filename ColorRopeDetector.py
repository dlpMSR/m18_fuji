import cv2
import numpy as np
import csv


class ColorRopeDetector(object):
    def __init__(self, img_bin, num_frame):
        self.img_bin = img_bin
        self.num_frame = num_frame
        self.height, self.width = img_bin.shape[:2]
        self.a_right, self.b_right = self._get_line()[0]
        self.a_left, self.b_left = self._get_line()[1]
        self.length_right, self.lengh_left = self._get_length()

    def _get_line(self):
        ROWS = [140, 180, 220, 260, 300, 700, 740, 780, 820, 860]
        border_right = []
        border_left = []
        for row in ROWS:
            scan_line = self.img_bin[row, :]
            scan_line = np.where(scan_line != 0)
            try:    
                i_right = np.max(scan_line)
                i_left = np.min(scan_line)
                border_right.append(i_right)
                border_left.append(i_left)
        A = np.array([ROWS, np.ones(len(ROWS))]).T
        result_right = np.linalg.lstsq(A, border_right, rcond=None)[0]
        result_left = np.linalg.lstsq(A, border_left, rcond=None)[0]
        return (result_right, result_left)

    def _get_length(self):
            length_R = int(self.width-self.b_left-(self.height/2)*self.a_left)
            length_L = int(self.b_right+(self.height/2)*self.a_right)
            return length_R, length_L

    def is_detected(self):
        THRESHOLD = 1
        if (self.a_right-self.a_left)**2 <THRESHOLD:
            return True
        else:
            return False

    def get_result(self):
        if self.is_detected():
            result_list = [self.num_frame, self.length_right, self.lengh_left]
        else:
            result_list = [self.num_frame]
        return result_list


class ImageProcessing(object):
    def __init__(self, img):
        self.img_origin = img

    def _get_color_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        LOWER_RED = np.array([200, 100, 100])
        UPPER_RED = np.array([310, 255, 255])
        img_mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
        return img_mask

    def _remove_noise(self, img_bin):
        labelStats = cv2.connectedComponentsWithStats(img_bin)
        masses = labelStats[2]
        for mass in masses:
            if mass[4] < 1000:
                cv2.rectangle(img_bin, (mass[0], mass[1]),
                              (mass[0] + mass[2], mass[1] + mass[3]), 0, -1)
        return img_bin
    
    def get_preprocessed_image(self):
        img = self._get_color_mask(self.img_origin)
        img = self._remove_noise(img)
        return img
    
    def get_output_image(self):
        img_output = self.img_origin
        return img_output


  