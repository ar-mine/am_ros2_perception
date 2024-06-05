import cv2
import numpy as np

import os
import time


class MarkerDetector:
    def __init__(self):
        # Aruco detector parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.aruco_size = 0.018

    def detect_markers(self, img, camera_matrix=None, dist_coeffs=None, square_length=2.9/100):
        img_input = img.copy()
        img_input_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        # Aruco markers detection
        marker_corners, marker_ids, rejected_candidates = self.detector.detectMarkers(img_input_gray)
        if len(marker_corners) > 0:
            cv2.aruco.drawDetectedMarkers(img_input, marker_corners, marker_ids)
            diamond_corners, diamond_ids = cv2.aruco.detectCharucoDiamond(img_input_gray,
                                                                          markerCorners=marker_corners,
                                                                          markerIds=marker_ids,
                                                                          squareMarkerLengthRate=200.0/180)
            if len(diamond_corners) > 0:
                cv2.aruco.drawDetectedDiamonds(img_input, diamond_corners, diamond_ids)
                if camera_matrix is not None and dist_coeffs is not None:
                    rvec, tvec, objPoints = cv2.aruco.estimatePoseSingleMarkers(diamond_corners, square_length,
                                                                                camera_matrix, dist_coeffs)
                    transform = []
                    for r, t in zip(rvec, tvec):
                        cv2.drawFrameAxes(img_input, camera_matrix, dist_coeffs, r, t, 0.10)  # axis length 50
                        R, _ = cv2.Rodrigues(r)
                        transformation_matrix = np.eye(4)
                        transformation_matrix[:3, :3] = R
                        transformation_matrix[:3, 3] = t
                        transform.append(transformation_matrix)
                    return img_input, transform
        return img_input

    def marker_generate(self, num, path):
        for marker_id in range(num):
            # Create charuco with diamond layout
            marker_image = cv2.aruco.drawCharucoDiamond(self.aruco_dict,
                                                        ids=(marker_id, marker_id, marker_id, marker_id),
                                                        squareLength=200, markerLength=180)

            # Save marker pictures
            cv2.imwrite(os.path.join(path, f'aruco_marker_{marker_id}.png'), marker_image)
            time.sleep(0.5)


def main():
    marker_detector = MarkerDetector()
    if 0:
        # Generate charuco_diamond images
        marker_detector.marker_generate(2, "/home/armine/Pictures/aruco")
    else:
        # Detect and show the pose on window
        cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，如果有多个摄像头，可以尝试不同的索引

        if not cap.isOpened():
            print("Cannot open camera!")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot get images!")
                break

            marked_img = marker_detector.detect_markers(frame)
            cv2.imshow('Camera', marked_img)

            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
