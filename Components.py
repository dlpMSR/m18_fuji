import cv2
import numpy as np


class DetectionProcessing(object):
    MEASURE_WIDTH_mm = 13
    MEASURE_WIDTH_pix = 39
    
    def __init__(self, img_bin):
        self.img_bin = img_bin
        self.height, self.width = img_bin.shape[:2]
        self.a_right = 0
        self.b_right = 0
        self.a_left = 0
        self.b_left = 0
        self.is_detected = self._seek_line()
        self.lengthR_pix = int(self.width-self.b_left-(self.height/2)*self.a_left)
        self.lengthL_pix = int(self.b_right+(self.height/2)*self.a_right)

    def _seek_line(self):
        ROWS = [140, 180, 220, 260, 300, 700, 740, 780, 820, 860]
        is_detected = True
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
            except:
                is_detected = False
                break
        if is_detected:        
            A = np.array([ROWS, np.ones(len(ROWS))]).T
            self.a_right, self.b_right = np.linalg.lstsq(A, border_right, rcond=None)[0]
            self.a_left, self.b_left = np.linalg.lstsq(A, border_left, rcond=None)[0]
        return is_detected
        
    def get_result(self):
        if self.is_detected:
            mmppix = DetectionProcessing.MEASURE_WIDTH_mm/DetectionProcessing.MEASURE_WIDTH_pix
            lengthR_mm = int(mmppix*self.lengthR_pix)
            lengthL_mm = int(mmppix*self.lengthL_pix)
            result_list = [self.lengthL_pix, self.lengthR_pix, lengthL_mm, lengthR_mm]
        else:
            result_list = []
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
    
    def get_output_image(self, frame_status):
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        img_output = self.img_origin
        if frame_status.is_detected:
            height = frame_status.height
            width = frame_status.width
            ar = frame_status.a_right
            br = frame_status.b_right
            al = frame_status.a_left
            bl = frame_status.b_left
            length_R = str(frame_status.lengthR_pix)
            length_L = str(frame_status.lengthL_pix)
            cv2.line(img_output, (int(br), 0), (int(br+height*ar), height), (0, 0, 255), 2)
            cv2.line(img_output, (int(bl), 0), (int(bl+height*al), height), (0, 0, 255), 2)
            cv2.line(img_output, (0, int(height/2)), (int(bl+(height/2)*al), int(height/2)), (255, 255, 0), 2)
            cv2.line(img_output, (width, int(height/2)), (int(br+(height/2)*ar), int(height/2)), (255, 0, 255), 2)
            cv2.putText(img_output, str(length_L), (int(width/8), int(height/2-10)), FONT, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img_output, str(length_R), (int(7*width/8), int(height/2-10)), FONT, 1, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            pass
        return img_output


  