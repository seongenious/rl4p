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
from typing import Any

import env.parking_env as env
from model.planner import RsPlanner
from model.autoencoder import AutoEncoder
from configs import EnvStatus, EnvConfig


def save_image(img: np.ndarray, img_path: Path, quality: int = 95) -> None:
    """Save image with optimized settings"""
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    # Save with JPEG compression
    cv2.imwrite(str(img_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


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
        mean, std = encoder.embed(img_tensor)
        return mean.squeeze(0).numpy()  # Return mean features


def generate_data(
    encoder: Any, 
    env: env.ParkingEnv, 
    num_episodes: int, 
    max_steps_per_episode: int
) -> List[Dict[str, Any]]:
    """Generate guardian training data"""
    data = []
    
    for episode_idx in tqdm(range(num_episodes), desc="Generating data..."):
        obs = env.reset(episode_idx + 1)
        done = False
        
        while not done:
            # Sample random action
            action = env.action_space.sample()
            delta, speed = action[0], action[1]
            
            # Take step
            obs, reward, status, info = env.step(action=action)
            
            # Extract BEV features
            img = obs['img']
            bev_features = extract_bev_features(encoder, img)
            
            # Store data
            data.append({
                'bev_features': bev_features,
                'action': action,
                'collision': (status == EnvStatus.COLLISION),
            })
            
            done = status != EnvStatus.RUNNING
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate guardian training data')
    parser.add_argument('--autoencoder_path', type=str, default="../model/autoencoder.pth",
                        help='Path to autoencoder model')
    parser.add_argument('--output_dir', type=str, default="../data/collision",
                        help='Output directory for collision data')
    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup config
    config = EnvConfig()
    config.render_rs_path = False
    config.render_info = False

    # Setup environment
    env = env.ParkingEnv(render_mode="rgb_array", config=config)

    # Load autoencoder
    autoencoder = load_autoencoder(args.autoencoder_path)

    num_episodes = min(args.num_episodes, 
                      len(parking_env.map.map_data) if parking_env.map.map_data else args.num_episodes)
    
    print(f"Generating guardian training data...")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Output directory: {output_dir.absolute()}")

    # Generate collision data
    collision_data = generate_guardian_data(autoencoder, parking_env, num_episodes, args.max_steps)
    
    # Save data
    output_file = output_dir / "guardian_training_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(collision_data, f)
    
    # Calculate statistics
    total_samples = len(collision_data)
    collision_samples = sum(1 for data in collision_data if data['is_collision'])
    collision_rate = collision_samples / total_samples if total_samples > 0 else 0
    
    print(f"\n=== Data Generation Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Collision samples: {collision_samples}")
    print(f"Collision rate: {collision_rate:.3f}")
    print(f"Data saved to: {output_file}")
    print(f"BEV feature dimension: {collision_data[0]['bev_features'].shape[0] if collision_data else 'N/A'}")
    
    parking_env.close()
    print("Done!")
