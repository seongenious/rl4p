import sys
sys.path.append("..")
sys.path.append(".")
import argparse

import torch
import numpy as np
from typing import Any

from env.parking_env import ParkingEnv
from model.planner import RsPlanner
from configs import EnvStatus, EnvConfig
from model.autoencoder import AutoEncoder
from model.guardian import Guardian


def load_encoder(path: str = '../ckpt/autoencoder/autoencoder.pt', device: torch.device = torch.device('cpu')):
    """Load pre-trained autoencoder model"""
    try:
        encoder = torch.load(path, map_location=device, weights_only=False)
        encoder.eval()
        encoder.to(device)
        return encoder
    except FileNotFoundError:
        raise FileNotFoundError(f"AutoEncoder model not found at {path}")


def load_guardian(path: str = '../ckpt/guardian/guardian.pt', device: torch.device = torch.device('cpu')):
    """Load pre-trained guardian model"""
    try:
        guardian = torch.load(path, map_location=device, weights_only=False)
        guardian.eval()
        guardian.to(device)
        return guardian
    except FileNotFoundError:
        raise FileNotFoundError(f"Guardian model not found at {path}")


def extract_bev_features(encoder: Any, img: np.ndarray, device: torch.device) -> np.ndarray:
    """Extract BEV features using autoencoder"""
    with torch.no_grad():
        # Convert numpy array to torch tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).float().to(device)
        else:
            img_tensor = img.float().to(device)
        
        # Add batch dimension if needed
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Normalize to [0, 1] if needed
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        
        # Extract features
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # HWC -> CHW
        mean, _ = encoder.embed(img_tensor)
        return mean


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = load_encoder(device=device)
    guardian = load_guardian(device=device)

    env = ParkingEnv(render_mode="human")

    step_distance = env.config.vehicle.step_time * env.config.vehicle.num_step * env.config.action.speed[1]
    planner = RsPlanner(step_distance)

    for i in range(10):
        obs = env.reset(i + 1)
        done = False
        while not done:
            img = obs['img']
            bev_feat = extract_bev_features(encoder, img, device)
            
            # Convert action to tensor and move to device
            action = env.action_space.sample()
            action_tensor = torch.from_numpy(action).float().to(device)
            
            # Get collision probability
            p_collide = guardian(bev_feat, action_tensor)
            env.risk = p_collide.detach().cpu().numpy()[0][0]

            next_obs, reward, status, info = env.step(action=action)
            done = status != EnvStatus.RUNNING

            if info['rs_path'] is not None:
                planner.set_path(info['rs_path'])
    
    env.close()