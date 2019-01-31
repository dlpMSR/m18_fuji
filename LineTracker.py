import numpy as np
import cv2
import csv
from tqdm import tqdm

from ColorRopeDetector import ColorRopeDetector
from ColorRopeDetector import ImageProcessing

def lineTracker(video_path):
    cap = cv2.VideoCapture(video_path)
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1440,1080))
    f = open('./output.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['frame', 'right','left'])
    result_list = []
    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = ImageProcessing(frame)
            frame_preprocessed = frame.get_preprocessed_image()
            frame_status = ColorRopeDetector(frame_preprocessed, frame_num)
            out.write(frame.get_output_image())
            result_list.append(frame_status.get_result())
            cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
            cv2.imshow('frame',frame.get_output_image())
            frame_num += 1
            pbar.update(1)
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


def main():
    lineTracker('./00011.MTS')


if __name__ == '__main__':
    main()  