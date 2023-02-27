from .trainer import BaseTrainer, CRDTrainer, RecordTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "custom": RecordTrainer
}
