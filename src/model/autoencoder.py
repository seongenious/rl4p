import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, pooling=2, batch_norm=True):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool2d(pooling),
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(pooling),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x) + self.residual(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample, padding=None, batch_norm=True):
        super().__init__()
        padding = padding or kernel_size // 2

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.UpsamplingBilinear2d(upsample),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

        self.residual = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(upsample),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x) + self.residual(x)

class ImgEncoder(nn.Module):
    def __init__(self, img_dim, kernel_size, embed_dim, conv_dims, fc_dims, pooling=2, padding=None, batch_norm=True):
        super().__init__()
        C, W, H = img_dim 

        conv_dims.insert(0, C)
        conv_layers = []
        for i in range(len(conv_dims) - 1):
            conv_layers.append(ConvBlock(conv_dims[i], conv_dims[i + 1], kernel_size, stride=1, padding=padding, pooling=pooling, batch_norm=batch_norm))
        
        fc_in_dim = int(W * H * conv_dims[-1] / (4 ** (len(conv_layers))))
        fc_dims.insert(0, fc_in_dim)
        fc_layers = []
        for i in range(len(fc_dims) - 1):
            fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1]))
            fc_layers.append(nn.Tanh())
        
        self.net = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            *fc_layers,
        )
        self.output_mean = nn.Linear(fc_dims[-1], embed_dim)
        self.output_std = nn.Linear(fc_dims[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        mean = self.output_mean(x)
        std = self.output_std(x)
        return mean, std

class ImgDecoder(nn.Module):
    def __init__(self, img_dim, kernel_size, embed_dim, conv_dims, fc_dims, padding=None, batch_norm=True):
        super().__init__()
        self.img_dim = img_dim
        self.conv_dims = conv_dims
        C, W, H = img_dim

        fc_dims.insert(0, embed_dim)
        fc_layers = []
        for i in range(len(fc_dims) - 1):
            fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1]))
            fc_layers.append(nn.Tanh())
        
        fc_out_dim = int(H / (2 ** len(conv_dims)) * W / (2 ** len(conv_dims)) * conv_dims[-1])
        fc_layers.append(nn.Linear(fc_dims[-1], fc_out_dim))
        self.fc_net = nn.Sequential(*fc_layers)

        conv_layers = []
        upsample_size = int(H / (2 ** len(conv_dims)) * 2)
        for i in range(len(conv_dims) - 1):
            conv_layers.append(DeconvBlock(conv_dims[-i - 1], conv_dims[-i - 2], kernel_size, upsample_size, padding, batch_norm))
            upsample_size *= 2
        conv_layers.append(DeconvBlock(conv_dims[0], C, kernel_size, upsample_size, padding, batch_norm))
        self.conv_net = nn.Sequential(*conv_layers)

        self.output = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_net(z)
        B = x.shape[0]
        width = int(self.img_dim[-2] / (2 ** len(self.conv_dims)))
        x = x.reshape((B, self.conv_dims[-1], width, width))
        x = self.conv_net(x)
        x = self.output(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, img_dim, kernel_size, embed_dim, conv_dims, fc_dims):
        super(AutoEncoder, self).__init__()
        self.encoder = ImgEncoder(img_dim, kernel_size, embed_dim, conv_dims, fc_dims)
        self.decoder = ImgDecoder(img_dim, kernel_size, embed_dim, conv_dims, fc_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean, _ = self.encoder(x)
        return self.decoder(mean)

    def embed(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            mean, std = self.encoder(img)
        return mean, std
  
    def save(self, path: str):
        torch.save(self, path)
        print(f"AutoEncoder saved to {path}")