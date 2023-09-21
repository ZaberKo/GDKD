from detectron2.config import get_cfg, CfgNode as CN, LazyCall as L


base_cfg = get_cfg()

base_cfg.MODEL.MOBILENETV2 = CN()
base_cfg.MODEL.MOBILENETV2.DEBUG = 0
base_cfg.MODEL.MOBILENETV2.OUT_FEATURES = ['m2']
base_cfg.MODEL.MOBILENETV2.NORM = 'FrozenBN'

# model_cfg = base_cfg.MODEL.clone()
# input_cfg = base_cfg.INPUT.clone()

cfg = base_cfg.clone()
cfg.TEACHER = base_cfg.clone()

# cfg.TEACHER=CN()
# cfg.TEACHER.MODEL = model_cfg.clone()
# cfg.TEACHER.INPUT = input_cfg.clone()
# cfg.STUDENT = CN()
# cfg.STUDENT.MODEL = model_cfg.clone()
# cfg.STUDENT.INPUT = input_cfg.clone()


cfg.EXPERIMENT = CN()
cfg.EXPERIMENT.PROJECT = "detection_coco"
cfg.EXPERIMENT.NAME = ""
cfg.EXPERIMENT.TAG = []
cfg.EXPERIMENT.WANDB = True

cfg.KD = CN()
cfg.KD.FEATURE_KD_MASK = None
cfg.KD.TYPE = "DKD"


cfg.KD.REVIEWKD = CN()
cfg.KD.REVIEWKD.KD_WEIGHT = 1.0
# ABF settings:
cfg.KD.REVIEWKD.IN_CHANNELS = [256, 256, 256, 256, 256]
cfg.KD.REVIEWKD.OUT_CHANNELS = [256, 256, 256, 256, 256]
cfg.KD.REVIEWKD.MAX_MID_CHANNEL = 256

cfg.KD.DKD = CN()
cfg.KD.DKD.ALPHA = 1.0
cfg.KD.DKD.BETA = 0.25
cfg.KD.DKD.T = 1.0
cfg.KD.DKD.DISTILL_TYPE =  "all" # all or things

cfg.KD.GDKD = CN()
cfg.KD.GDKD.TOPK = 5
cfg.KD.GDKD.W0 = 1.0
cfg.KD.GDKD.W1 = 0.125
cfg.KD.GDKD.W2 = 0.25
cfg.KD.GDKD.T = 1.0
cfg.KD.GDKD.DISTILL_TYPE = "all"
cfg.KD.GDKD.WARMUP = 36000 # unit: iters

def get_distiller_config():
    return cfg.clone()

    