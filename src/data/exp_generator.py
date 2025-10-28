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

import env.parking_env as env
from model.planner import RsPlanner
from model.autoencoder import AutoEncoder
from configs import EnvStatus, EnvConfig


def save_image(img, img_path, quality=95):
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


def load_autoencoder(model_path="../model/autoencoder.pth"):
    """Load pre-trained autoencoder model"""
    try:
        autoencoder = torch.load(model_path, map_location='cpu')
        autoencoder.eval()
        print(f"AutoEncoder loaded from {model_path}")
        return autoencoder
    except FileNotFoundError:
        print(f"AutoEncoder model not found at {model_path}")
        print("Creating a new AutoEncoder model...")
        # Create a new autoencoder with default parameters
        autoencoder = AutoEncoder(
            img_dim=(3, 64, 64),
            kernel_size=3,
            embed_dim=128,
            conv_dims=[4, 8, 16],
            fc_dims=[256]
        )
        autoencoder.eval()
        return autoencoder


def extract_bev_features(autoencoder, img):
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
        mean, std = autoencoder.embed(img_tensor)
        return mean.squeeze(0).numpy()  # Return mean features


def generate_guardian_data(autoencoder, env, num_episodes, max_steps_per_episode):
    """Generate guardian training data"""
    collision_data = []
    
    for episode_idx in tqdm(range(num_episodes), desc="Generating collision data"):
        obs = env.reset(episode_idx + 1)
        done = False
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            # Sample random action
            action = env.action_space.sample()
            delta, speed = action[0], action[1]
            
            # Take step
            obs, reward, status, info = env.step(action=action)
            
            # Check collision
            is_collision = (status == EnvStatus.COLLISION)
            
            # Extract BEV features
            img = obs['img']
            bev_features = extract_bev_features(autoencoder, img)
            
            # Store data
            data_point = {
                'bev_features': bev_features,
                'delta': delta,
                'speed': speed,
                'is_collision': is_collision,
                'episode': episode_idx,
                'step': step_count,
                'status': status.value
            }
            collision_data.append(data_point)
            
            step_count += 1
            done = status != EnvStatus.RUNNING
    
    return collision_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate guardian training data')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to generate')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum steps per episode')
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
    parking_env = env.ParkingEnv(render_mode="rgb_array", config=config)

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
