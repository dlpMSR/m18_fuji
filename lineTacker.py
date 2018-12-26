import numpy as np
import cv2


def generateFrameImage():
    cap = cv2.VideoCapture('./03.mp4')
    num = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        filename = './frame/{}.jpg'.format(num)
        cv2.imwrite(filename, frame)
        num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def imageProcessing():
    frame = cv2.imread('./frame/114.jpg')
    height, width = frame.shape[:2]

    # 色マスク
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    lowerRed = np.array([30, 50, 150])
    upperRed = np.array([120, 255, 255])
    img_mask = cv2.inRange(hsv, lowerRed, upperRed)
    # ノイズキャンセリング
    labelStats = cv2.connectedComponentsWithStats(img_mask)
    nLabels, labelImages, masses, center = labelStats
    for mass in masses:
        # 閾値50はてきとー
        if mass[4] < 50:
            cv2.rectangle(img_mask, (mass[0], mass[1]),
                          (mass[0] + mass[2], mass[1] + mass[3]), 0, -1)
    # 線のヘリを探してくる
    border_list = []
    rows = [360, 400, 440, 480, 520, 560, 600, 640, 680, 720]
    for row in rows:
        line = img_mask[row, :]
        for i in range(len(line)):
            if line[i] != 0:
                break
        border = (row, i)
        border_list.append(border)
    print(border_list)

    #cv2.imwrite('./mask.jpg', img_mask)
    #img_color = cv2.bitwise_and(frame, frame, mask=img_mask)


def remove_noise_by_labeling(binary_image, value, threshold):
    stats_of_connectedcomponents = cv2.connectedComponentsWithStats(
        binary_image)
    bounding_boxes = np.delete(stats_of_connectedcomponents[2], 0, 0)
    for box in bounding_boxes:
        if box[value] < threshold:
            cv2.rectangle(binary_image, (box[0], box[1]),
                          (box[0]+box[2], box[1]+box[3]), 0, -1)
    return binary_image

    output = cv2.resize(img_mask, (round(width / 4), round(height / 4)))
    imshow(output)
    #cv2.imwrite('./mask.jpg', img_mask)


def filter(img):
    kernel_3x3 = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], np.float32)
    img_filtered = cv2.filter2D(img_mask, -1, kernel_gradient_3x3)
    return img_filtered


def imshow(img):
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    imageProcessing()
    # generateFrameImage()


if __name__ == '__main__':
    main()
