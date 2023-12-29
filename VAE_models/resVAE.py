import torch
import torch.nn as nn

class ResModel(torch.nn.Module):
    def __init__(self, path_to_weights=None, bottleneck_size=16, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        super().__init__() 
        self.bottleneck_size = bottleneck_size
        self.encoder = Encoder(bottleneck_size=self.bottleneck_size)
        self.decoder = Decoder(bottleneck_size=self.bottleneck_size)
        self.prior_distribution = torch.distributions.Normal(0, 1)
        #use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if path_to_weights is not None:
            self.load_state_dict(torch.load(path_to_weights, map_location=device)["model"])
        self.to(device)

    def forward(self,
                target_data=None,  # during training this should be a batch of target data.
                # During inference simply leave this out to sample unconditionally.
                noise_scale_during_inference=1.4,
                z=None):
        if target_data is not None:
            # run the encoder
            means, variance = self.encoder(target_data)
            variance = variance.exp()  # so that our model learns to predict in log space, which has more room to work with
            # convert means and variance to latent sample
            z = means + variance * self.prior_distribution.sample(means.shape).to(self.device)
        else:
            if z is None:
                z = torch.randn(self.bottleneck_size).to(self.device).unsqueeze(0) * noise_scale_during_inference
        # run the decoder
        reconstructions_of_targets = self.decoder(z)

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x + identity

class Encoder(torch.nn.Module):
    def __init__(self, bottleneck_size):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(64, 16, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.resnet_blocks = torch.nn.Sequential(
            ResNetBlock(16, 16),
            ResNetBlock(16, 16),
            ResNetBlock(16, 32),
            ResNetBlock(32, 32),
            ResNetBlock(32, 32),
            ResNetBlock(32, 16)
        )
        self.proj_mean = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, bottleneck_size),
            torch.nn.Tanh(),
            torch.nn.Linear(bottleneck_size, bottleneck_size),
            torch.nn.ReLU(),
        )
        self.proj_var = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, bottleneck_size),
            torch.nn.Tanh(),
            torch.nn.Linear(bottleneck_size, bottleneck_size),
            torch.nn.ReLU(),
        )

    def forward(self, target_data_for_compression):
        x = self.conv1(target_data_for_compression)
        x = self.relu(x)
        compressed = self.resnet_blocks(x)
        return self.proj_mean(compressed), self.proj_var(compressed)


class Decoder(torch.nn.Module):
    def __init__(self, bottleneck_size):
        super(Decoder, self).__init__()
        self.resnet_blocks = torch.nn.Sequential(
            ResNetBlock(16, 16),
            ResNetBlock(16, 16),
            ResNetBlock(16, 32),
            ResNetBlock(32, 32),
            ResNetBlock(32, 32),
            ResNetBlock(32, 64)
        )
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 64),
            torch.nn.Tanh()
        )

    def forward(self, compressed_data_for_decompression):
        decompressed = self.resnet_blocks(compressed_data_for_decompression)
        return self.proj(decompressed)
