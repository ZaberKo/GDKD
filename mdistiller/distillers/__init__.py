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
from .GDKD import GDKD
from .DKDMod import DKDMod
from .DIST import DIST

from ..models import get_model

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
    "DIST": DIST
}


def get_distiller(cfg, **kwargs):
    model_student = get_model(cfg, cfg.DISTILLER.STUDENT, pretrained=False)

    if cfg.DISTILLER.TYPE == "NONE":
        distiller = Vanilla(model_student)
    else:
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
