import sys
sys.path.append("..")
sys.path.append(".")
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np

import env.parking_env as env
from model.planner import RsPlanner
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate BEV images for autoencoder training')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality (0-100)')
    args = parser.parse_args()

    # Setup path
    img_dir = Path("../data/img")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Setup environment and paths
    config = EnvConfig()
    env = env.ParkingEnv(render_mode="rgb_array", config=config)

    # Setup planner
    step_distance = env.config.vehicle.step_time * env.config.vehicle.num_step * env.config.action.speed[1]
    planner = RsPlanner(step_distance)

    num_episodes = 0 if env.map.map_data is None else len(env.map.map_data)
    max_steps = env.config.max_time_step
    total_images = 0

    # Case 1: Random action
    for idx in tqdm(range(num_episodes), desc="Random episodes"):
        obs = env.reset(idx + 1)
        done = False
        step_count = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, status, info = env.step(action=action)
            done = status != EnvStatus.RUNNING

            # Save image
            img = obs['img']
            img_path = img_dir / f'random_ep{idx:04d}_step{step_count:04d}.jpg'
            save_image(img, img_path, quality=args.quality)
            
            step_count += 1
            total_images += 1

    # Case 2: Planner action    
    for idx in tqdm(range(num_episodes), desc="Planner episodes"):
        obs = env.reset(idx + 1)
        done = False
        step_count = 0
        
        while not done:
            action = planner.get_action()
            obs, reward, status, info = env.step(action=action)
            done = status != EnvStatus.RUNNING

            if info['rs_path'] is not None:
                planner.set_path(info['rs_path'])

            # Save image
            img = obs['img']
            img_path = img_dir / f'planner_ep{idx:04d}_step{step_count:04d}.jpg'
            save_image(img, img_path, quality=args.quality)
            
            step_count += 1
            total_images += 1
    
    env.close()
    
    print(f'Successfully generated {total_images} images.')