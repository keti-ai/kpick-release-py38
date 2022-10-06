from kpick.dataset.cornell import CornellDatasetGui

class CornellDatasetPickGui(CornellDatasetGui):
    pass





def demo_cornell_gui(cfg_path=None):
    from ketisdk.gui.gui import GuiModule, GUI
    module = GuiModule(CornellDatasetPickGui, name='Cornell', cfg_path=cfg_path, num_method=3)
    GUI(title='Cornell GUI', modules=[module,], data_root='data/cornell',
        default_cfg_path='kpick/apps/configs/cornell_dataset.cfg',
        rgb_formats=['*/*r.png'], depth_formats=['*/*d.tiff'])

if __name__=='__main__':
    demo_cornell_gui()