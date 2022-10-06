from ketisdk.vision.base.base_objects import BasObj

class BasDetect():
    """ basic detection process

    :param args: configuration arguments
    :type args: :class:`.CFG`
    """
    # def __init__(self,args=None):
    #     self.args=args

    def get_model(self):
        """ get model """
        self.model=None
        print('Model loaded ...')

    def detect(self, **kwargs):
        """ predict """
        print('Detecting ...')


class BasDetectObj(BasObj, BasDetect):
    """ [:class:`.BasObj`, :class:`.BasDetect`] basic detection obj

        :param args: configuration arguments
        :type args: :class:`.CFG`
        :param cfg_path: configuation file path
        :type cfg_path: .cfg file
        """
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        super().__init__(args=args, cfg_path=cfg_path, name=name, default_args=default_args)
        self.get_model()

    def reload_params(self):
        super().reload_params()
        if hasattr(self, 'model'):
            if hasattr(self.model,'reload_params'): self.model.reload_params()

class DetGui():
    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        pass
    def finalize_acc(self):
        pass
    def init_acc(self):
        pass

class GuiObj(BasObj, DetGui):
    pass

class DetGuiObj(BasDetectObj, DetGui):
    pass