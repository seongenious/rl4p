import os

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self):
        self.writer.close()
