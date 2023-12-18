from yacs.config import CfgNode as CN
from mdistiller.engine.cfg import CFG

from .SKD import SKD
from .SGDKD import SGDKD
from .ADKD import ADKD
from .GDKDAutok import GDKDAutok
from .GDKD_perclass_k import GDKDPerClassK
from .GDKDAutoW import GDKDAutoW
from .ReviewKD_GDKD import ReviewKD_GDKD
from .DIST_GDKD import DIST_GDKD

from mdistiller.distillers import distiller_dict

distiller_dict.update({
    "SGDKD": SGDKD,
    "ADKD": ADKD,
    "SKD": SKD,
    "GDKDAutok": GDKDAutok,
    "GDKDPerClassK": GDKDPerClassK,
    "GDKDAutoW": GDKDAutoW,
    "REVIEWKD_GDKD": ReviewKD_GDKD,
    "DIST_GDKD": DIST_GDKD
})


CFG.ADKD = CN()
CFG.ADKD.CE_WEIGHT = 1.0
CFG.ADKD.ALPHA = 1.0
CFG.ADKD.BASE_GAMMA = 1.0
CFG.ADKD.TARGET_GAMMA = 0.1
CFG.ADKD.T = 4.0
CFG.ADKD.WARMUP = 20
CFG.ADKD.KL_TYPE = "forward"

CFG.SGDKD = CN()
CFG.SGDKD.CE_WEIGHT = 1.0
CFG.SGDKD.W0 = 1.0
CFG.SGDKD.W1 = 8.0
CFG.SGDKD.ETA = 0.75
CFG.SGDKD.KL_TYPE = "forward"
CFG.SGDKD.T = 4.0
CFG.SGDKD.WARMUP = 20

CFG.SKD = CN()
CFG.SKD.CE_WEIGHT = 1.0
CFG.SKD.KD_WEIGHT = 8.0
CFG.SKD.ETA = 0.5
CFG.SKD.KL_TYPE = "forward"
CFG.SKD.T = 4.0
CFG.SKD.WARMUP = 20

CFG.GDKDAUTOK = CN()
CFG.GDKDAUTOK.CE_WEIGHT = 1.0
CFG.GDKDAUTOK.W0 = 1.0
CFG.GDKDAUTOK.W1 = 4.0
CFG.GDKDAUTOK.W2 = 8.0
CFG.GDKDAUTOK.TOPK_TH = 100 # 100 means no limit in cifar100
CFG.GDKDAUTOK.RATIO_TH = 2.0
CFG.GDKDAUTOK.KL_TYPE = "forward"
CFG.GDKDAUTOK.T = 4.0
CFG.GDKDAUTOK.WARMUP = 20

CFG.GDKDPerClassK = CN()
CFG.GDKDPerClassK.CE_WEIGHT = 1.0
CFG.GDKDPerClassK.W0 = 1.0
CFG.GDKDPerClassK.W1 = 4.0
CFG.GDKDPerClassK.W2 = 8.0
CFG.GDKDPerClassK.PREBUILD_TOPK_TH = 100 # 100 means no limit in cifar100
CFG.GDKDPerClassK.PREBUILD_RATIO_TH = 2.0
CFG.GDKDPerClassK.PRELOAD_TOPK_PATH = None # eg: "./debug_topk.yaml"
CFG.GDKDPerClassK.KL_TYPE = "forward"
CFG.GDKDPerClassK.T = 4.0
CFG.GDKDPerClassK.WARMUP = 20

CFG.GDKDAutoW = CN()
CFG.GDKDAutoW.CE_WEIGHT = 1.0
CFG.GDKDAutoW.M1 = 1.0
CFG.GDKDAutoW.M2 = 1.0
CFG.GDKDAutoW.W1 = 1.0
CFG.GDKDAutoW.W2 = 1.0
CFG.GDKDAutoW.MODE = "v1"
CFG.GDKDAutoW.TOPK = 5
CFG.GDKDAutoW.T = 4.0
CFG.GDKDAutoW.WARMUP = 20
