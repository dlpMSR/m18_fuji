import numpy as np
import cv2
import csv
from tqdm import tqdm


def lineTracker():
    cap = cv2.VideoCapture('./00011.MTS')
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm(total=int(length))
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
            frame, result = imageProcessing(frame)
            out.write(frame)
            cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
            cv2.imshow('frame',frame)
            result_list.append([frame_num]+result)
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

def generateFrameImage():
    cap = cv2.VideoCapture('./00011.MTS')
    num = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:    
            img_mask = verification(frame)
            cv2.imshow('frame', img_mask)
            filename_f = './frame/{}.jpg'.format(num)
            filename_m = './mask/{}.jpg'.format(num)
            cv2.imwrite(filename_f, frame)
            cv2.imwrite(filename_m, img_mask)
            num += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def imageProcessing(frame):
    height, width = frame.shape[:2]
    # 色マスク
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    LOWER_RED = np.array([200, 100, 100])
    UPPER_RED = np.array([310, 255, 255])
    img_mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    # ノイズキャンセリング
    labelStats = cv2.connectedComponentsWithStats(img_mask)
    masses = labelStats[2]
    for mass in masses:
        if mass[4] < 1000:
            cv2.rectangle(img_mask, (mass[0], mass[1]),
                          (mass[0] + mass[2], mass[1] + mass[3]), 0, -1)
    # 線のヘリを探してくる
    border_right = []
    border_left = []
    rows = [140, 180, 220, 260, 300, 700, 740, 780, 820, 860]
    for row in rows:
        line = img_mask[row, :]
        for i_right in range(len(line)):
            if line[i_right] != 0:
                break
        for i_left in reversed(range(len(line))):
            if line[i_left] != 0:
                break
        border_right.append(i_right)
        border_left.append(i_left)
    #最小二乗法を敢行
    X = rows
    A = np.array([X, np.ones(len(X))])
    A =A.T
    Y_right = border_right
    Y_left = border_left
    ar, br = np.linalg.lstsq(A, Y_right, rcond=None)[0]
    al, bl = np.linalg.lstsq(A, Y_left, rcond=None)[0]

    length_R = int(width-bl-int(height/2)*al)
    length_L = int(br+int(height/2)*ar)
    #書き込み
    cv2.line(frame, (int(br), 0), (int(br+height*ar), height), (0, 0, 255), 3)
    cv2.line(frame, (int(bl), 0), (int(bl+height*al), height), (0, 0, 255), 3)
    cv2.line(frame, (0, int(height/2)), (int(br+int(height/2)*ar), int(height/2)), (0, 255, 0), 2)
    cv2.line(frame, (width, int(height/2)), (int(bl+int(height/2)*al), int(height/2)), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(length_L), (int(width/4), int(height/2-10)), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, str(length_R), (int(3*width/4), int(height/2-10)), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
    #csvに書きこみ
    result = [length_L, length_R]
    return frame, result

def verification(frame):
    height, width = frame.shape[:2]
    # 色マスク
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    lowerGreen = np.array([200, 100, 100])
    upperGreen = np.array([310, 255, 255])
    img_mask = cv2.inRange(hsv, lowerGreen, upperGreen)
    # ノイズキャンセリング
    labelStats = cv2.connectedComponentsWithStats(img_mask)
    masses = labelStats[2]
    for mass in masses:
        if mass[4] < 1000:
            cv2.rectangle(img_mask, (mass[0], mass[1]),
                          (mass[0] + mass[2], mass[1] + mass[3]), 0, -1)
    return img_mask

def main():
    lineTracker()
    # imageProcessing()
    # generateFrameImage()


if __name__ == '__main__':
    main()
