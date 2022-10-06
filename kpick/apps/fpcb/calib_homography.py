import cv2
import numpy as np

def Convert_Camera2Robot(CameraPosx, CameraPosy):
    RobotPosx = homographyMatrix[0][0][0] * CameraPosx + homographyMatrix[0][0][1] * CameraPosy + \
                homographyMatrix[0][0][2]
    RobotPosy = homographyMatrix[0][1][0] * CameraPosx + homographyMatrix[0][1][1] * CameraPosy + \
                homographyMatrix[0][1][2]
    RobotPosz = homographyMatrix[0][2][0] * CameraPosx + homographyMatrix[0][2][1] * CameraPosy + \
                homographyMatrix[0][2][2]

    Robot_Posx = RobotPosx / RobotPosz
    Robot_Posy = RobotPosy / RobotPosz

    return Robot_Posx, Robot_Posy

# Priority
# val_out = ([0.57, 0.97, 0.93, 0.87, 0.71, 0.36], [(1269, 122), (981, 428), (1131, 385), (82, 118), (1051, 380), (1200, 555)], [180.0, 90.0, 180.0, 180.0, 180.0, 90.0], [50.0, 108.0, 61.0, 66.0, 123.0, 22.0])
val_out = ([0.84, 0.64], [(642, 341), (786, 420)], [180.0, 180.0], [24.0, 26.0])

scores = val_out[0]
scores_max = max(scores)
scores_sel = scores.index(scores_max)
grip_centers = val_out[1][scores_sel]
x = grip_centers[0]
y = grip_centers[1]

print("Picking Point(Image) : ", x, y)

# Calibration
# 1.fing homography
cameraPoints = np.array([[319, 191], [512, 191], [512, 318], [319, 318]])
robotPoints = np.array([[571.91, 50.64], [574.02, -249.52], [374.56, -248.89], [374.47, 50.65]])

homographyMatrix = cv2.findHomography(cameraPoints, robotPoints)

# 2. convert coordinate
robotX, robotY = Convert_Camera2Robot(x, y)
print('Robot Coordinate', robotX, robotY)