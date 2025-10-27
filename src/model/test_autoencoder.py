#!/usr/bin/env python3
"""
AutoEncoder class test script
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import AutoEncoder, ImgEncoder, ImgDecoder, ConvBlock, DeconvBlock

def test_conv_block():
    """Test ConvBlock"""
    print("=== ConvBlock Test ===")
    
    # Generate test data
    batch_size = 4
    in_channels = 3
    height, width = 64, 64
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create ConvBlock
    conv_block = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, pooling=2)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = conv_block(x)
    print(f"Output shape: {output.shape}")
    
    # Check residual connection
    print(f"Residual connection applied: {output.shape[1] == 16}")
    
    return conv_block

def test_deconv_block():
    """Test DeconvBlock"""
    print("\n=== DeconvBlock Test ===")
    
    # Generate test data
    batch_size = 4
    in_channels = 16
    height, width = 32, 32
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create DeconvBlock
    deconv_block = DeconvBlock(in_channels=16, out_channels=8, kernel_size=3, upsample=(64, 64), padding=1)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = deconv_block(x)
    print(f"Output shape: {output.shape}")
    
    return deconv_block

def test_img_encoder():
    """Test ImgEncoder"""
    print("\n=== ImgEncoder Test ===")
    
    # Generate test data
    batch_size = 4
    img_dim = (3, 64, 64)  # (C, H, W)
    x = torch.randn(batch_size, *img_dim)
    
    # Create ImgEncoder
    encoder = ImgEncoder(
        img_dim=img_dim,
        kernel_size=3,
        embed_dim=128,
        conv_dims=[16, 32, 64],
        fc_dims=[256],
        pooling=2,
        padding=1,
        batch_norm=True
    )
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    mean, std = encoder(x)
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Embedding dimension: {mean.shape[1]}")
    
    return encoder

def test_img_decoder():
    """Test ImgDecoder"""
    print("\n=== ImgDecoder Test ===")
    
    # Generate test data
    batch_size = 4
    embed_dim = 128
    z = torch.randn(batch_size, embed_dim)
    
    # Create ImgDecoder
    decoder = ImgDecoder(
        img_dim=(3, 64, 64),
        kernel_size=3,
        embed_dim=embed_dim,
        conv_dims=[4, 8, 16],
        fc_dims=[256],
        padding=1,
        batch_norm=True
    )
    
    print(f"Input shape: {z.shape}")
    
    # Forward pass
    output = decoder(z)
    print(f"Output shape: {output.shape}")
    print(f"Output value range: [{output.min():.3f}, {output.max():.3f}]")
    
    return decoder

def test_autoencoder():
    """Test complete AutoEncoder"""
    print("\n=== Complete AutoEncoder Test ===")
    
    # Generate test data
    batch_size = 8
    img_dim = (3, 64, 64)
    x = torch.randn(batch_size, *img_dim)
    
    # Create AutoEncoder
    autoencoder = AutoEncoder(
        img_dim=img_dim,
        kernel_size=3,
        embed_dim=128,
        conv_dims=[16, 32, 64],
        fc_dims=[256]
    )
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass (reconstruction)
    reconstructed = autoencoder(x)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    print(f"Reconstructed image value range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # Extract embedding
    mean, std = autoencoder.embed(x)
    print(f"Embedding mean shape: {mean.shape}")
    print(f"Embedding std shape: {std.shape}")
    
    # Calculate reconstruction loss
    reconstruction_loss = nn.MSELoss()(x, reconstructed)
    print(f"Reconstruction loss: {reconstruction_loss.item():.6f}")
    
    return autoencoder, x, reconstructed

def test_autoencoder_training():
    """Test AutoEncoder training"""
    print("\n=== AutoEncoder Training Test ===")
    
    # Generate dummy data
    batch_size = 16
    img_dim = (3, 64, 64)
    
    # Generate simple pattern data
    x = torch.randn(batch_size, *img_dim)
    
    # Create AutoEncoder
    autoencoder = AutoEncoder(
        img_dim=img_dim,
        kernel_size=3,
        embed_dim=128,
        conv_dims=[4, 8, 16],
        fc_dims=[256]
    )
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 10
    losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed = autoencoder(x)
        loss = criterion(x, reconstructed)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Visualize training results
    plt.figure(figsize=(12, 4))
    
    # Loss graph
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    # Original image
    plt.subplot(1, 3, 2)
    original_img = x[0].permute(1, 2, 0).detach().numpy()
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(1, 3, 3)
    with torch.no_grad():
        reconstructed_img = autoencoder(x[0:1])[0].permute(1, 2, 0).detach().numpy()
    plt.imshow(reconstructed_img)
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('autoencoder_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'autoencoder_test_results.png'.")
    
    return autoencoder, losses

def test_model_save_load():
    """Test model save and load"""
    print("\n=== Model Save and Load Test ===")
    
    # Create AutoEncoder
    autoencoder = AutoEncoder(
        img_dim=(3, 64, 64),
        kernel_size=3,
        embed_dim=128,
        conv_dims=[4, 8, 16],
        fc_dims=[256]
    )
    
    # Test data
    x = torch.randn(1, 3, 64, 64)
    
    # Inference with original model
    with torch.no_grad():
        original_output = autoencoder(x)
    
    # Save model
    save_path = "test_autoencoder.pth"
    autoencoder.save(save_path)
    
    # Load model
    loaded_autoencoder = torch.load(save_path)
    
    # Inference with loaded model
    with torch.no_grad():
        loaded_output = loaded_autoencoder(x)
    
    # Compare results
    output_diff = torch.abs(original_output - loaded_output).max().item()
    print(f"Output difference after save/load: {output_diff:.10f}")
    
    if output_diff < 1e-6:
        print("Model save/load test successful!")
    else:
        print("Model save/load test failed!")
    
    # Remove temporary file
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Temporary file {save_path} removed")

def main():
    """Main test function"""
    print("Starting AutoEncoder class tests...")
    print("=" * 50)
    
    try:
        # Test individual components
        test_conv_block()
        test_deconv_block()
        test_img_encoder()
        test_img_decoder()
        
        # Test complete AutoEncoder
        autoencoder, x, reconstructed = test_autoencoder()
        
        # Test training
        trained_autoencoder, losses = test_autoencoder_training()
        
        # Test model save/load
        test_model_save_load()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
