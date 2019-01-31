import sys
import numpy as np
import cv2
import csv
from tqdm import tqdm

from Components import DetectionProcessing
from Components import ImageProcessing


def lineTracker(video_path):
    cap = cv2.VideoCapture(video_path)
    filename = video_path[:-4]
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('{}_output.avi'.format(filename),fourcc, 30.0, (1440,1080))
    f = open('{}_output.csv'.format(filename), 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['num', 'L_pix', 'R_pix', 'L_mm','R_mm'])
    result_list = []
    num_frame = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            img = ImageProcessing(frame)
            img_preprocessed = img.get_preprocessed_image()
            frame_status = DetectionProcessing(img_preprocessed)
            img_output = img.get_output_image(frame_status)
            out.write(img_output)
            if frame_status.is_detected:
                result_list.append([num_frame]+frame_status.get_result())
            num_frame += 1
            pbar.update(1)
            cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
            cv2.imshow('frame', img_output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    for item in result_list:
        writer.writerow(item)
    pbar.close()
    f.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main(arg):
    lineTracker(arg)


if __name__ == '__main__':
    args = sys.argv
    main(args[1])  