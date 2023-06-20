import cv2


def main():
    cal_round_point_calibration_plate("round_point-20X_NA0.40-2880x2048.bmp")


# Calculate the resolution scale of the circle in the round point calibration plate
def cal_round_point_calibration_plate(image_path: str):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100, param1=190, param2=30, minRadius=150, maxRadius=300)

    for x, y, r in circles[0]:
        cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 2, cv2.LINE_AA)
        print(int(x), int(y))
        print(int(r))
        print("\n")
    cv2.imshow('circle', image)
    cv2.waitKey(0)

    cv2.destroyWindow('circle')


if __name__ == "__main__":
    main()
