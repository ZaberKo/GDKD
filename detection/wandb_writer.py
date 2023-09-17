import wandb

from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.config import CfgNode

class WandbWriter(EventWriter):
    def __init__(self, cfg: CfgNode, window_size=20) -> None:
        self.window_size = window_size



    def write(self):
        storage = get_event_storage()


        for k, (v, record_iter) in storage.latest_with_smoothing_hint(self.window_size).items():
            wandb.log({k: v}, step=storage.iter, commit=False)
        
        wandb.log({})

            
    