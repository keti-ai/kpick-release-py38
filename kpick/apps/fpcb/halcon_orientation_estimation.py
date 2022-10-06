import halcon as ha
import math
import cv2
import os

def SM_get_model(modelPath, roi):
    '''
        roi: left, top, right, bottom
    '''
    ModelImage = ha.read_image(modelPath)
    left, top, right, bottom = roi
    ROI  = ha.gen_rectangle1(row_1=top, column_1=left, row_2=bottom, column_2=right)
    ImageROI = ha.reduce_domain(ModelImage, ROI)

    ShapeModelImages, ShapeModelRegions = ha.inspect_shape_model(ImageROI, 8,30)
    AreaModelRegions, RowModelRegions, ColumnModelRegions = ha.area_center(ShapeModelRegions)
    HeightPyramid = ha.count_obj(ShapeModelRegions)

    NumLevels = None
    for i in range(HeightPyramid):
       if AreaModelRegions[i] >= 15:
           NumLevels = i+1
    
    ModelID = ha.create_shape_model (ImageROI, NumLevels, 0, 2*math.pi, 'auto', 'none', 'use_polarity', 30, 10)
    ShapeModel = ha.get_shape_model_contours (ModelID, 1)
    
    return ModelID, ShapeModel

def SM_match(imagePath, ModelID, ShapeModel):
    SearchImage = ha.read_image(imagePath)
    RowCheck, ColumnCheck, AngleCheck, Score = ha.find_shape_model (SearchImage, ModelID, 0, 2*math.pi, 0.7, 1, 0.5, 'least_squares', 0, 0.7)
    
    AngleCheck = [el*180/math.pi for el in AngleCheck]

    print(f'Detected Objects: {AngleCheck} degrees')

def DM_get_model(imagePath=None, roi=None,CamParam=None, WorldX=None, WorldY=None):
    '''
        roi: left, top, right, bottom
    '''
    modelPath = 'data/apps/fpcb/model/dm_model.dsm'
    if imagePath is None:
        ModelID = ha.read_descriptor_model (modelPath)
        return ModelID
    
    # Image = ha.read_image(imagePath)
    im = cv2.imread(imagePath)
    left, top, right, bottom = roi

    im = cv2.resize(im, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('im',im[top:bottom, left:right, :])
    # # cv2.imshow('im',im)
    # cv2.waitKey()
    Image = ha.himage_from_numpy_array(im)
    
    Rectangle  = ha.gen_rectangle1(row_1=top, column_1=left, row_2=bottom, column_2=right)
    ImageReduced = ha.reduce_domain(Image, Rectangle)

    RowsRoi = [top,top,bottom, bottom]
    ColumnsRoi = [left, right, right, left]
    Pose, Quality = ha.vector_to_pose (WorldX, WorldY, [], RowsRoi, ColumnsRoi, CamParam, 'iterative', 'error')



    ModelID = ha.create_calib_descriptor_model (ImageReduced, CamParam, Pose, 'harris_binomial', [], [], 
                                                ['depth','number_ferns','patch_size','min_scale','max_scale'], [11,30,17,0.4,1.2], 42)
    
    ha.write_descriptor_model(ModelID, modelPath)
    return ModelID
    
def DM_match(imagePath, ModelID, CamParam):

    im = cv2.imread(imagePath)
    # im = cv2.rotate(im, cv2.ROTATE_180)
    im = cv2.resize(im, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
    Image = ha.himage_from_numpy_array(im)
    
    Pose, Score = ha.find_calib_descriptor_model (Image, ModelID, [], [], [], [], 0.25, 1, CamParam, 'num_points')

    print(f'Pose: {Pose}')


def demo_shape_matching():
    modelPath = 'data/apps/fpcb/board/board-01.png'
    roi = [188, 182, 298, 412]
    imagePath = 'data/apps/fpcb/board/board-09.png'

    # modelPath = 'data/apps/fpcb/testset1/000.png'
    # roi = [615, 285, 860, 620]
    # imagePath = 'data/apps/fpcb/testset1/000.png'
    
    ModelID, ShapeModel = SM_get_model(modelPath=modelPath, roi=roi)
    SM_match(imagePath=imagePath, ModelID=ModelID, ShapeModel=ShapeModel)

def demo_descriptor_matching():
    CamParam = [0.0155565, -109.42, 1.28008e-005, 1.28e-005, 322.78, 240.31, 640, 480]
    WorldX = [-0.189,0.189,0.189,-0.189]    
    WorldY = [-0.080,-0.080,0.080,0.080] 

    # modelImagePath = 'data/apps/fpcb/packaging/cookie_box_01.png'
    # # modelImagePath = None
    # roi = [115, 224, 540, 406]
    # imagePath = 'data/apps/fpcb/packaging/cookie_box_01.png'

    modelImagePath = 'data/apps/fpcb/testset1/000.png'
    # modelImagePath = None
    roi = [305, 190, 435, 410]
    # roi = [305, 190, 360, 288]
    imagePath = 'data/apps/fpcb/testset1/004.png'


    ModelID = DM_get_model(imagePath=modelImagePath, roi=roi, CamParam=CamParam, WorldX=WorldX, WorldY=WorldY)
    DM_match(imagePath=imagePath, ModelID=ModelID, CamParam=CamParam)

def demo_polar_transform():
    imagePath = 'data/apps/fpcb/testset1/001.png'
    im = cv2.imread(imagePath)

    # im = cv2.resize(im, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
    h,w = im.shape[:2]
    # xc, yc = w//2, h//2
    xc, yc = (670,350)
    out = im.copy()
    cv2.drawMarker(out, (xc, yc), (0,0,255), cv2.MARKER_TILTED_CROSS, 10, 1)
    cv2.circle(out,(xc,yc), 100, (0,255,0))
    cv2.imshow('im', out)
    cv2.waitKey()




if __name__=='__main__':
    demo_polar_transform()

    



    