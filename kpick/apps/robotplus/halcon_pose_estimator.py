import halcon  as ha
from ketisdk.gui.default_config import default_args
from ketisdk.utils.proc_utils import CFG, WorkSpace, ProcUtils
from ketisdk.vision.utils.rgbd_utils_v2 import RGBD
import cv2
import numpy as np
import os
import json

def get_default_args():
    BASE_CFGS = default_args()
    
    BASE_CFGS.net = CFG()
    BASE_CFGS.net.model_dir = 'data/apps/robotplus/model'
    BASE_CFGS.net.denoise_ksize = 25
    BASE_CFGS.net.thresh =0.001
    BASE_CFGS.net.ref_ind = 0

    BASE_CFGS.sensor.cam_intr = {'ppx':645.937, 'ppy': 347.56, 'fx': 916.963, 'fy': 916.676}

    return BASE_CFGS



class HaSurfaceMatcher():

    def get_model(self, modelDir, cam_intr, denoise_ksize):
        workspace_file = os.path.join(modelDir, 'workspace.json')
        if not os.path.exists(workspace_file):


            print(f'{workspace_file} does not exist -> EXIT')
            return
        with open(workspace_file, 'r') as f:
            pts = json.load(f)
        self.workspace = WorkSpace(pts=pts)
        print('workspace loaded')
        
        bg_rgb_file = os.path.join(modelDir, 'bg_rgb.png')
        bg_depth_file = os.path.join(modelDir, 'bg_depth.png')
        if not os.path.exists(bg_depth_file):
            print(f'{bg_depth_file} does not exist -> EXIT')
            return
        self.bg_rgbd = RGBD(rgb=cv2.imread(bg_rgb_file)[:,:,::-1], depth=cv2.imread(bg_depth_file, cv2.IMREAD_UNCHANGED), workspace=self.workspace)
        print('background loaded')

        self.surface_rgbds = []
        for i in range(10):
            surface_rgb_file = os.path.join(modelDir, f'surface{i}_rgb.png')
            surface_depth_file = os.path.join(modelDir, f'surface{i}_depth.png')
            if not os.path.exists(surface_depth_file):
                break
            self.surface_rgbds.append(RGBD(rgb=cv2.imread(surface_rgb_file)[:,:,::-1], 
                        depth=cv2.imread(surface_depth_file, cv2.IMREAD_UNCHANGED), workspace=self.workspace))
        print(f'{len(self.surface_rgbds)} surfaces loaded')


        self.update_model(cam_intr, denoise_ksize)

    
    def update_model(self, cam_intr, denoise_ksize, force_update=False, workspace=None, save_dir='data/model'):
        bg_xyz_file = os.path.join(self.args.net.model_dir, f'bg_xyz_{denoise_ksize}.npy')
        if workspace is not None:
            ws_file = os.path.join(save_dir, 'workspace.json')
            with open(ws_file, 'w') as outfile:
                outfile.write(json.dumps(self.workspace.pts.tolist()))
            print(f'New workspace saved to {ws_file}')

        if not os.path.exists(bg_xyz_file) or force_update:
            if workspace is not None:
                self.bg_rgbd.set_workspace(pts=workspace.pts)
            bg_xyz = self.bg_rgbd.crop_xyz(intr_params=cam_intr, denoiseKSize=denoise_ksize)
            np.save(bg_xyz_file, bg_xyz)
        else:
            bg_xyz = np.load(bg_xyz_file)
            print(f'bg_xyz loaded from {bg_xyz_file}')
        
        surface_xyzs = []
        for j,rgbd in enumerate(self.surface_rgbds):
            surface_xyz_file = os.path.join(self.args.net.model_dir, f'surface{j}_xyz_{denoise_ksize}.npy')
            if not os.path.exists(surface_xyz_file) or force_update:
                if workspace is not None:
                    rgbd.set_workspace(pts=workspace.pts)
                surface_xyz = rgbd.crop_xyz(intr_params=cam_intr, denoiseKSize=denoise_ksize)
                np.save(surface_xyz_file, surface_xyz)
            else:
                surface_xyz = np.load(surface_xyz_file)
                print(f'surface_xyz loaded from {surface_xyz_file}')
            surface_xyzs.append(surface_xyz)
        
        self.SFMs, self.SFM_centers, self.SFM_radiuss = [],[], []
        for surface_xyz in surface_xyzs:
            SFM, SFM_center, radius = self.get_a_surface_model(bg_xyz, surface_xyz)
            self.SFMs.append(SFM)
            self.SFM_centers.append(SFM_center)
            self.SFM_radiuss.append(radius)
        print(f'{len(self.SFMs)} surface models updated')      

        self.SFM_colors = ProcUtils().get_color_list(len(self.SFMs))


    def get_a_surface_model(self, bg_xyz, surface_xyz):
        bg_Image = ha.himage_from_numpy_array(bg_xyz)
        XRef, YRef, ZRef = ha.decompose3(bg_Image)
        self.ZRef = ZRef
        # ReferenceScene = ha.xyz_to_object_model_3d (XRef, YRef, ZRef)

        surface_Image = ha.himage_from_numpy_array(surface_xyz)
        Xm, Ym, Zm = ha.decompose3(surface_Image)
        ImageSub = ha.sub_image(ZRef, Zm, 1, 0)
        Region = ha.threshold(ImageSub, 10,1e+10)
        ConnectedRegions = ha.connection(Region)
        SelectedRegions = ha.select_shape(ConnectedRegions, 'area', 'and', 100, 1e+10)
        RegionUnion = ha.union1(SelectedRegions)
        # ObjectModel3DReduced = ha.reduce_object_model_3d_by_view (RegionUnion, ObjectModel3D, [], [])
        Xm = ha.reduce_domain (Xm, RegionUnion)
        ObjectModel3DModel = ha.xyz_to_object_model_3d (Xm, Ym, Zm)
        
        #
        SFM = ha.create_surface_model (ObjectModel3DModel, 0.03, [], [])

        center = ha.get_surface_model_param(SFM, 'center')
        radius = ha.get_surface_model_param(SFM, 'bounding_box1')[-3:]
        print(f'center: {center}, radius: {radius}')
        

        return SFM, center, radius

    def matchSurface(self, rgbd, cam_intr,ref_ind=None, denoiseMedianRadius=25,thresh=None, model_dir='data/model'): 
        XYZ = rgbd.crop_xyz(intr_params=cam_intr, denoiseKSize=denoiseMedianRadius)
        Image = ha.himage_from_numpy_array(XYZ)
        X,Y,Z = ha.decompose3(Image)
        ha.write_image(Image, format='tiff',fill_color=0, file_name=os.path.join(self.args.net.model_dir, 'test0_xyz'))

        wr, hr = ha.get_image_size(self.ZRef)
        w,h = ha.get_image_size(Z)

        if ((wr[0], hr[0]) != (w[0], h[0])):
            self.update_model(cam_intr=cam_intr, denoise_ksize=denoiseMedianRadius, force_update=True, workspace=rgbd.workspace, save_dir=model_dir)
 
        ImageSub = ha.sub_image (self.ZRef, Z, 1, 0)
        Region = ha.threshold (ImageSub, 12, 1e+10)
        XReduced = ha.reduce_domain (X, Region)        
        ObjectModel3DSceneReduced = ha.xyz_to_object_model_3d (XReduced, Y, Z)
        Target_SFM_ = ha.create_surface_model (ObjectModel3DSceneReduced, 0.03, [], [])
        Taget_center = ha.get_surface_model_param(Target_SFM_, 'center')


        PoseOut = []
        for i, SFM in enumerate(self.SFMs):
            if ref_ind is not None:
                if i!=ref_ind:
                    continue
            try:
                Pose, Score, SurfaceMatchingResultID = ha.find_surface_model (SFM, ObjectModel3DSceneReduced, 0.05, 0.3, 0.2, 'true', 'num_matches', 10)
            except:
                Pose, Score = [],[]
            if len(Score)==0:
                continue
            sx, sy, sz = self.SFM_centers[i]
            # sx, sy, sz = 0,0,0
            for j, score in enumerate(Score):
                if thresh is not None:
                    if score < thresh:
                        continue
                CPose = Pose[7*j: 7*(j+1)]
                # RigidTrans = ha.rigid_trans_object_model_3d(self.Surface3DModels[i], CPose)
                x,y, z, thetax, thetay, thetaz = Pose[7*j: 7*(j+1)-1]
                x,y,z  = Taget_center
                PoseOut.append([x,y,z, thetax, thetay, thetaz, i, score])
                ha.himage_as_numpy_array

        return PoseOut

    def xyzToPixel(self, x,y,z, cam_intr):
        isDict = isinstance(cam_intr, dict)
        fx = cam_intr['fx'] if isDict else cam_intr.fx
        fy = cam_intr['fy'] if isDict else cam_intr.fy
        xc = cam_intr['ppx'] if isDict else cam_intr.ppx
        yc = cam_intr['ppy'] if isDict else cam_intr.ppy

        px, py = int(x*fx/z + xc), int(y*fy/z + yc)
        return (px, py)

    def project3DTo2D(self, Loc3D, cam_intr):
        isDict = isinstance(cam_intr, dict)
        fx = cam_intr['fx'] if isDict else cam_intr.fx
        fy = cam_intr['fy'] if isDict else cam_intr.fy
        ppx = cam_intr['ppx'] if isDict else cam_intr.ppx
        ppy = cam_intr['ppy'] if isDict else cam_intr.ppy

        X, Y, Z = Loc3D[:,0], Loc3D[:,1], Loc3D[:,2]

        Ix, Iy  = np.divide(X,Z)*fx + ppx, np.divide(Y, Z)*fy + ppy
        X3 =  np.concatenate((Ix.reshape(-1, 1), Iy.reshape(-1, 1)), axis=1)
        
        return X3.astype('int')

    def getAffine3D(self, trans, theta):
        xc, yc, zc = trans
        theX, theY, theZ = theta
        T = np.array([[1,0,0, xc],
                        [0,1, 0, yc],
                        [0,0,1, zc],
                        [0,0,0,1] 
                        ])
        cosx, sinx = np.cos(theX/180*np.pi), np.sin(theX/180*np.pi)
        Rx = np.array([[1,0,0,0],
                        [0, cosx, sinx, 0],
                        [0, -sinx, cosx, 0],
                        [0,0,0,1]
                        ])
        cosy, siny = np.cos(theY/180*np.pi), np.sin(theY/180*np.pi)
        Ry = np.array([[cosy, 0, -siny, 0],
                        [0, 1, 0,0],
                        [siny, 0, cosy, 0],
                        [0,0,0,1]
                        ])
        cosz, sinz = np.cos(theZ/180*np.pi), np.sin(theZ/180*np.pi)
        Rz = np.array([[cosz, -sinz, 0, 0],
                        [sinz, cosz, 0,0],
                        [0, 0, 1, 0],
                        [0,0,0,1]
                        ])

        M =  np.dot(T, np.dot(Rz, np.dot(Ry, Rx)))
        return M


    def calcBBox3D(self, center, radius, theta):

        rx,ry,rz = radius
        X1 = np.array([[-rx, -ry, -rz, 1],
                        [rx, -ry, -rz, 1],
                        [rx, ry,-rz, 1],
                        [-rx, ry, -rz, 1], 
                        [-rx, -ry, rz, 1],
                        [rx, -ry, rz, 1],
                        [rx, ry,rz, 1],     
                        [-rx, ry, rz, 1], 
                        [0,0,0,1]           # center
                        ]).transpose([1,0])
        # X2 = np.dot(T, np.dot(Rz, np.dot(Ry, np.dot(Rx, X1))))
        
        M  = self.getAffine3D(center, theta)
        X2 = np.dot(M, X1)
        
        return X2[:3, :].transpose([1,0])

    def drawBox(self, im, bbox, color):
        out = np.copy(im)

        bb1 = bbox[:4, :]
        bb2 = np.roll(bb1, 1, axis=0)
        for pt1, pt2, in zip(bb1, bb2):
            cv2.line(out, tuple(pt1), tuple(pt2), color, 2)
        bb1 = bbox[4:8, :]
        bb2 = np.roll(bb1, 1, axis=0)
        for pt1, pt2, in zip(bb1, bb2):
            cv2.line(out, tuple(pt1), tuple(pt2), color, 2)
        bb1 = bbox[:4, :]
        bb2 = bbox[4:8, :]
        for pt1, pt2, in zip(bb1, bb2):
            cv2.line(out, tuple(pt1), tuple(pt2), color, 2)
        center = tuple(bbox[-1, :])
        cv2.drawMarker(out, center, color,cv2.MARKER_TILTED_CROSS, 7, 2)

        return out


    def calcAxes3D(self,center, theta, unit_len=30):
        M = self.getAffine3D(center, theta)
        rr = unit_len
        X1 = np.array([[rr, 0, 0, 1],
                        [0, rr,0, 1],
                        [0,0,rr, 1],
                        [0,0,0,1]           # center
                        ]).transpose([1,0])
        X2 = np.dot(M, X1)
        return X2[:3, :].transpose([1,0])

    def drawAxes2D(self, im, axes2D):
        out = np.copy(im)
        org = tuple(axes2D[-1,:])
        for pt, color in zip(axes2D[:3,:], [(255, 0, 0), (0,255,0), (0,0,255)]): 
            cv2.line(out, org, tuple(pt), color, 2)
        return out


from kpick.base.base import DetGuiObj
class HaSurfaceMatcherGui(HaSurfaceMatcher, DetGuiObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed'):
        DetGuiObj.__init__(self, args=args, cfg_path=cfg_path, name=name, default_args=get_default_args())
    
    def get_model(self):
        HaSurfaceMatcher.get_model(self, modelDir=self.args.net.model_dir, 
                    cam_intr=self.args.sensor.cam_intr, denoise_ksize=self.args.net.denoise_ksize)

    def matchSurfaceAndShow(self, rgbd, cam_intr, disp_mode='rgb'):
        out = rgbd.disp(mode=disp_mode)
        Pose = self.matchSurface(rgbd, cam_intr, denoiseMedianRadius= self.args.net.denoise_ksize,  
                        thresh=self.args.net.thresh, ref_ind=self.args.net.ref_ind, model_dir=self.args.net.model_dir)
        print(Pose)

        for j, pose in enumerate(Pose):
            x, y = self.xyzToPixel(pose[0], pose[1], pose[2], self.args.sensor.cam_intr)
            sur_ind = pose[-2]
            # print(f'Detected object pixel: {(x,y)}')
            # cv2.putText(out, f'{j}', (x,y), cv2.FONT_HERSHEY_COMPLEX, self.args.disp.text_scale, self.SFM_colors[sur_ind],self.args.disp.text_thick)
            bbox3D = self.calcBBox3D(pose[:3], self.SFM_radiuss[sur_ind], pose[3:6], )
            bbox2D = self.project3DTo2D(bbox3D, self.args.sensor.cam_intr)
            out = self.drawBox(out, bbox2D, self.SFM_colors[sur_ind])
            
            axes3D = self.calcAxes3D(center=pose[:3], theta=pose[3:6])
            axes2D = self.project3DTo2D(Loc3D=axes3D, cam_intr=self.args.sensor.cam_intr)
            out = self.drawAxes2D(out, axes2D)
        
        return {'im': out}

    def register_model(self, sensor, workspace):
        save_dir = self.args.net.model_dir
        os.makedirs(save_dir, exist_ok=True)
        

        s = input('To save BACKGROUND: REMOVE all items and ENTER [Q to EXIT]')
        if s in ['q', 'Q']:
            return
        self.bg_rgbd = sensor.get_rgbd(workspace=workspace)
        self.workspace = workspace
        with open(os.path.join(save_dir, 'workspace.json'), 'w') as outfile:
            outfile.write(json.dumps(self.workspace.pts.tolist()))
        print('workspace saved')

        cv2.imwrite(os.path.join(save_dir, 'bg_rgb.png'), self.bg_rgbd.bgr())
        cv2.imwrite(os.path.join(save_dir, 'bg_depth.png'), self.bg_rgbd.depth)

        bg_xyz = self.bg_rgbd.crop_xyz(self.args.sensor.cam_intr, self.args.net.denoise_ksize)
        bg_Image = ha.himage_from_numpy_array(bg_xyz)
        ha.write_image(bg_Image, format='tiff',fill_color=0, file_name=os.path.join(save_dir, 'bg_xyz'))
        
        print('background saved')

        self.surface_rgbds = []
        while True:
            s = input('To save a SURFACE: PLACE a item and ENTER [Q to EXIT]')
            if s in ['q', 'Q']:
                break
            self.surface_rgbds.append(sensor.get_rgbd(workspace=workspace))
        
        for j, rgbd in enumerate(self.surface_rgbds):
            cv2.imwrite(os.path.join(save_dir, f'surface{j}_rgb.png'), rgbd.bgr())
            cv2.imwrite(os.path.join(save_dir, f'surface{j}_depth.png'), rgbd.depth)
            xyz = rgbd.crop_xyz(self.args.sensor.cam_intr, self.args.net.denoise_ksize)
            Image = ha.himage_from_numpy_array(xyz)
            ha.write_image(Image, format='tiff',fill_color=0, file_name=os.path.join(save_dir, f'surface{j}_xyz'))

        self.update_model(self.args.sensor.cam_intr, self.args.net.denoise_ksize, force_update=True)

        print('Registration completed...')

        


    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind==0:
            ret = self.matchSurfaceAndShow(rgbd, self.args.sensor.cam_intr, disp_mode)
        if method_ind==1:
            ret = self.register_model(sensor=kwargs['sensor'], workspace=kwargs['workspace'])
        return ret


def demo_halcon_surface_matching(cfg_path=None, default_cfg_path='configs/default.cfg', data_root=None, rgb_formats=None, depth_formats=None):
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.sensor.zivid_sensor import get_zivid_module
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module0 = GuiModule(HaSurfaceMatcherGui, type='ha_pose', name='Ha Poser',
                              category='detector', cfg_path=cfg_path, num_method=3, 
                              key_args=['net.denoise_ksize', 'net.thresh', 'net.ref_ind'])

    GUI(title='Halcon Surface Matching',
           modules=[detect_module0] + get_realsense_modules() + [get_zivid_module(),],
           default_cfg_path=default_cfg_path,
           data_root=data_root, rgb_formats=rgb_formats,  depth_formats=depth_formats,
           )

if __name__=='__main__':
    # HaSurfaceMatcherGui()
    demo_halcon_surface_matching(default_cfg_path='kpick/apps/configs/halcon_default.cfg', cfg_path='kpick/apps/configs/oi_object.cfg',
                                data_root='data/apps/robotplus/OI_object/220816', rgb_formats=['*rgb*',], depth_formats=['*depth*',])




    
        

        