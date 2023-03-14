from .trainer import BaseTrainer, CRDTrainer, RecordTrainer, CRDRecordTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "custom": RecordTrainer,
    "custom_crd": CRDRecordTrainer,
}
