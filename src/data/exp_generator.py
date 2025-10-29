import sys
sys.path.append("..")
sys.path.append(".")
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle
import torch
from typing import Any, List, Dict

from env.parking_env import ParkingEnv
from model.autoencoder import AutoEncoder
from configs import EnvStatus, EnvConfig


def load_encoder(path: str = '../model/autoencoder.pth'):
    """Load pre-trained autoencoder model"""
    try:
        encoder = torch.load(path, map_location='cpu')
        encoder.eval()
        return encoder
    except FileNotFoundError:
        raise FileNotFoundError(f"AutoEncoder model not found at {path}")


def extract_bev_features(encoder: Any, img: np.ndarray) -> np.ndarray:
    """Extract BEV features using autoencoder"""
    with torch.no_grad():
        # Convert numpy array to torch tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).float()
        else:
            img_tensor = img.float()
        
        # Add batch dimension if needed
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Normalize to [0, 1] if needed
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        
        # Extract features
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # HWC -> CHW
        mean, _ = encoder.embed(img_tensor)
        return mean.squeeze(0).numpy()  # Return mean features


def generate_data(encoder: Any) -> List[Dict[str, Any]]:
    """Generate guardian training data"""
    # Setup config
    config = EnvConfig()
    config.render_rs_path = False
    config.render_info = False

    # Setup environment
    env = ParkingEnv(render_mode="rgb_array", config=config)

    data = []
    for episode_idx in tqdm(range(len(env.map.map_data)), desc="Generating data..."):
        obs = env.reset(episode_idx + 1)
        done = False
        
        while not done:
            action = env.action_space.sample()
            next_obs, reward, status, info = env.step(action=action)
            
            # Extract BEV features
            img = obs['img']
            bev_features = extract_bev_features(encoder, img)
            
            # Store data
            data.append({
                'bev_features': bev_features,
                'action': action,
                'collision': (status == EnvStatus.COLLISION),
            })
            
            obs = next_obs
            done = status != EnvStatus.RUNNING
    
    env.close()
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate guardian training data')
    parser.add_argument('--autoencoder_path', type=str, default="../ckpt/autoencoder/autoencoder.pt",
                        help='Path to autoencoder model')
    parser.add_argument('--output_dir', type=str, default="../data/collision",
                        help='Output directory for collision data')
    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load autoencoder
    autoencoder = load_encoder(args.autoencoder_path)
    
    # Generate collision data
    data = generate_data(autoencoder)
    
    # Save data
    output_file = output_dir / "data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    # Calculate statistics
    total_samples = len(data)
    collision_samples = sum(1 for data in data if data['collision'])
    
    print(f"\n=== Data Generation Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Collision samples: {collision_samples}")
    print(f"Data saved to: {output_file}")
    
    
    print("Done!")
