from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .DKDMod import DKDMod
from .GDKD import GDKD
from .GDKD3 import GDKD3
from .DIST import DIST

from ..models import get_model
from mdistiller.engine.utils import load_checkpoint

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "DKDMod": DKDMod,
    "GDKD": GDKD,
    "GDKD3": GDKD3,
    "DIST": DIST,
}


def get_distiller(cfg, **kwargs):
    model_student = get_model(cfg, cfg.DISTILLER.STUDENT, pretrained=False)

    if cfg.DISTILLER.TYPE == "NONE":
        distiller = Vanilla(model_student)
    else:
        if cfg.DISTILLER.TEACHER_WEIGHTS:
            model_teacher = get_model(cfg, cfg.DISTILLER.TEACHER, pretrained=False)
            model_teacher.load_state_dict(
                load_checkpoint(cfg.DISTILLER.TEACHER_WEIGHTS), strict=True
            )
        else:
            # use default teacher weights
            model_teacher = get_model(cfg, cfg.DISTILLER.TEACHER, pretrained=True)

        if cfg.DISTILLER.TYPE == "CRD":
            distiller = CRD(
                model_student, model_teacher, cfg, num_data=kwargs["num_data"]
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    return distiller
