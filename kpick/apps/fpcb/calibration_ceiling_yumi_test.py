#coding:utf-8
import numpy as np
import json
import csv
class CvtCam2Rot():
    def __init__(self):
        self.offset=[0,0,0]
        with open("apps/fpcb/parameter/yumi_intrinsic_param.json", "r") as st_json:
            intrinsic_param = json.load(st_json)
        self.camera_matrix=np.array(intrinsic_param["cameraMatrix"])
        with open("apps/fpcb/parameter/yumi_affine.json", "r") as st_json:
            affine = json.load(st_json)
        self.affine=np.array(affine["affine"])

    def Convert_Camera2Robot3d(self,CameraPosx, CameraPosy, CameraDepthZ, affine3dMatrix):
        Robot_Posx = affine3dMatrix[0][0] * CameraPosx + affine3dMatrix[0][1] * CameraPosy + \
                     affine3dMatrix[0][2] * CameraDepthZ + affine3dMatrix[0][3]

        Robot_Posy = affine3dMatrix[1][0] * CameraPosx + affine3dMatrix[1][1] * CameraPosy + \
                     affine3dMatrix[1][2] * CameraDepthZ + affine3dMatrix[1][3]

        Robot_Posz = affine3dMatrix[2][0] * CameraPosx + affine3dMatrix[2][1] * CameraPosy + \
                     affine3dMatrix[2][2] * CameraDepthZ + affine3dMatrix[2][3]

        return Robot_Posx, Robot_Posy, Robot_Posz

    def Func(self, camera_x_2d, camera_y_2d, camera_z):
        points_c = [(camera_x_2d - self.camera_matrix[0][2]) / self.camera_matrix[0][0] * camera_z,
                    (camera_y_2d - self.camera_matrix[1][2]) / self.camera_matrix[1][1] * camera_z,
                    camera_z]
        points_c = np.array(points_c)
        calRobotX, calRobotY, calRobotZ = self.Convert_Camera2Robot3d(points_c[0], points_c[1], points_c[2],
                                                                      self.affine)
        calRobotX = float(calRobotX - self.offset[0])
        calRobotY = float(calRobotY - self.offset[1])
        calRobotZ = float(calRobotZ - self.offset[2])
        return [calRobotX, calRobotY, calRobotZ]
if __name__ == "__main__":
    print("cal validate")
    cam_cvt_robot=CvtCam2Rot()
    f = open("apps/fpcb/parameter/yumi_point.csv", 'r', encoding='utf-8')
    rdr = csv.reader(f)
    points_r = []  # robot
    points_c = []  # camera
    data_count = 0

    # read csv
    count=0
    for line in rdr:
        print("index: ", count)
        camera2d_x = float(line[3])
        camera2d_y = float(line[4])
        camera2d_z = float(line[5])

        robot3d=cam_cvt_robot.Func(camera2d_x,camera2d_y,camera2d_z)
        robot3d_init=[float(line[0]),float(line[1]),float(line[2])]

        print(robot3d_init,robot3d)
        print(np.array(robot3d_init)-np.array(robot3d))
        count+=1
