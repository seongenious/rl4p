import cv2
import numpy as np

from configs import EnvConfig
    
class ImageProcessor():
    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.downsample_rate = self.config.observation.img_downsample_rate
        self.n_channels = self.config.observation.img_channels

    def process_img(self, img):
        processed_img = cv2.resize(
          img, 
          (img.shape[0] // self.downsample_rate, img.shape[1] // self.downsample_rate),
        )
        processed_img = processed_img / 255.0

        return processed_img
    