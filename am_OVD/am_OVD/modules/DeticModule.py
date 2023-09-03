import sys
import os
script_path = os.path.dirname(os.path.abspath(__file__))
detic_path = os.path.join(script_path, 'Detic/')
centernet2_path = os.path.join(script_path, 'Detic/third_party/CenterNet2/')
sys.path.insert(0, detic_path)
sys.path.insert(0, centernet2_path)
from .Detic.demo import VisualizationDemo, setup_cfg


class DeticArgs:
    def __init__(self):
        self.confidence_threshold = 0.3
        self.config_file = os.path.join(detic_path, 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
        self.cpu = False
        # self.input = ['desk.jpg']
        self.opts = ['MODEL.WEIGHTS',
                     os.path.join(detic_path, 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth')]
        # self.output = 'out2.jpg'
        self.pred_all_class = False
        self.video_input = None
        self.vocabulary = 'lvis'
        # self.custom_vocabulary = 'hat,paper'
        self.webcam = None


class DeticModule:
    def __init__(self):
        args = DeticArgs()
        cfg = setup_cfg(args)

        cfg.defrost()
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(detic_path, cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
        for i in range(len(cfg.MODEL.TEST_CLASSIFIERS)):
            cfg.MODEL.TEST_CLASSIFIERS[i] = os.path.join(detic_path, cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH[i])
        cfg.freeze()

        self.detector = VisualizationDemo(cfg, args)

    def process(self, img):
        predictions, visualized_output = self.detector.run_on_image(img)
        return visualized_output.get_image()[:, :, ::-1]
