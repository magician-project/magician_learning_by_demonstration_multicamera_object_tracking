import cv2
import numpy as np
import glob
import os

def detect_aruco_from_pnm(path):
    # Load the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Camera intrinsic parameters (example values, replace with actual calibration)
    camera_matrix = np.array([[800, 0, 320],
                               [0, 800, 240],
                               [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))  # Assuming no distortion

    # Process all .pnm files in the given directory
    for file in glob.glob(os.path.join(path, "*.pnm")):
        print(f"Processing {file}...")
        frame = cv2.imread(file)
        if frame is None:
            print(f"Could not read {file}")
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
                #cv2.aruco.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.05)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.05)
        
        cv2.imshow(f"ArUco Detection - {os.path.basename(file)}", frame)
        cv2.imwrite("%s_out.pnm" % file, frame)

        cv2.waitKey(1)  # Wait for key press before proceeding
        cv2.destroyAllWindows()

# Example usage: change 'path_to_pnm_files' to your actual directory
path_to_pnm_files = "/home/ammar/Documents/Programming/Magician/src/c/grabber/QR"  # Change this to the actual path

detect_aruco_from_pnm(path_to_pnm_files)
