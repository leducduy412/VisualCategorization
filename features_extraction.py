import cv2
import numpy as np


def extract_features(images):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_all = []
    descriptors_all = []

    for image in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect Harris corners
        harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)

        # Normalize and threshold the corner strengths
        harris_corners = cv2.normalize(harris_corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32FC1)
        ret, harris_corners = cv2.threshold(harris_corners, 0.01 * harris_corners.max(), 255, cv2.THRESH_BINARY)
        harris_corners = np.uint8(harris_corners)

        # Detect keypoints with SIFT on the entire image
        keypoints = sift.detect(gray, None)

        # Filter keypoints using the binary Harris corners mask
        keypoints = [kp for kp in keypoints if harris_corners[int(kp.pt[1]), int(kp.pt[0])]]

        # Compute SIFT descriptors for the filtered keypoints
        keypoints, descriptors = sift.compute(gray, keypoints)

        keypoints_all.append(keypoints)
        descriptors_all.append(descriptors)

    return keypoints_all, descriptors_all


def extract_features_one(image):
    sift = cv2.xfeatures2d.SIFT_create()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect Harris corners
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Normalize and threshold the corner strengths
    harris_corners = cv2.normalize(harris_corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32FC1)
    ret, harris_corners = cv2.threshold(harris_corners, 0.01 * harris_corners.max(), 255, cv2.THRESH_BINARY)
    harris_corners = np.uint8(harris_corners)

    # Detect keypoints with SIFT on the entire image
    keypoints = sift.detect(gray, None)

    # Filter keypoints using the binary Harris corners mask
    keypoints = [kp for kp in keypoints if harris_corners[int(kp.pt[1]), int(kp.pt[0])]]

    # Compute SIFT descriptors for the filtered keypoints
    keypoints, descriptors = sift.compute(gray, keypoints)

    return descriptors
