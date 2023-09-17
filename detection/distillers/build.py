from detectron2.utils.registry import Registry

KD_REGISTRY = Registry("KD")
KD_REGISTRY.__doc__ = """
Registry for KD modules, eg: KD, DKD, GDKD, ReviewKD
"""

