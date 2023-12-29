import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import robust_loss_pytorch
from NVAE.utils import Swish, ChannelSELayer1d, reparameterize, kl, kl_2
from NVAE.blocks import EncoderResidualBlock, DecoderResidualBlock, ResidualBlock, ConvBlock, UpsamplingConvBlock, DownsamplingConvBlock

class EncoderBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels 
        modules = []
        modules.append(DownsamplingConvBlock(channels[0], channels[1]))
        self.module_list = nn.ModuleList(modules)
    
    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            #print(f'EncoderBlock {i+1} output shape:', x.shape)
        return x


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock([z_dim, z_dim]),
            EncoderBlock([z_dim, z_dim]),
            EncoderBlock([z_dim, z_dim]),
            EncoderBlock([z_dim, z_dim]),
            EncoderBlock([z_dim, z_dim]),
            EncoderBlock([z_dim, z_dim]),
        ])

        self.encoder_residual_blocks = nn.ModuleList([
            EncoderResidualBlock(z_dim),
            EncoderResidualBlock(z_dim),
            EncoderResidualBlock(z_dim),
            EncoderResidualBlock(z_dim),
            EncoderResidualBlock(z_dim),
            EncoderResidualBlock(z_dim),
        ])
        self.condition_x = nn.Sequential(
            Swish(),
            nn.Conv1d(z_dim, 2, kernel_size=3, padding=1) #C_out=2: generate the mean and log variance values
        )
        
    def forward(self, x):
        test = []
        xs = []
        last_x = x
        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x = r(e(x))
            last_x = x # last feature map
            xs.append(x)
        mu, log_var = self.condition_x(last_x).split(1, dim=1)
        return mu, log_var, xs[:-1][::-1] #drop last element & reverse the order of the feature maps (for decoder)

class DecoderBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        modules.append(UpsamplingConvBlock(channels[0], channels[1]))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            #print(f'DecoderBlock {i+1} output shape:', x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        # Input channels = z_channels * 2 = x_channels + z_channels
        # Output channels = z_channels
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2])
        ])
        self.decoder_residual_blocks = nn.ModuleList([
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim)
        ])
        
        # p(z_l | z_(l-1))
        self.condition_z = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, 2, kernel_size=1) #(z_dim, z_dim, kernel_size=1)
            )
        ])
        
        #self.condition_z_2 = nn.ModuleList([
        #    nn.Sequential(
        #        ResidualBlock(z_dim),
        #        Swish(),
        #        nn.Conv1d(z_dim, z_dim, kernel_size=1)
        #    ),
        #    nn.Sequential(
        #        ResidualBlock(z_dim),
        #        Swish(),
        #        nn.Conv1d(z_dim, 2, kernel_size=1) #C_out=2 for splitting into 2(mu,log_var)
        #    )
        #])

        # p(z_l | x, z_(l-1))
        self.condition_xz = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 2, kernel_size=1),
                Swish(),
                nn.Conv1d(2, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 2, kernel_size=1),
                Swish(),
                nn.Conv1d(2, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 2, kernel_size=1),
                Swish(),
                nn.Conv1d(2, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 2, kernel_size=1),
                Swish(),
                nn.Conv1d(2, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 2, kernel_size=1),
                Swish(),
                nn.Conv1d(2, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 2, kernel_size=1),
                Swish(),
                nn.Conv1d(2, 2, kernel_size=1)
            )
        ])

        self.recon = nn.Sequential(
            ResidualBlock(z_dim),
            nn.Conv1d(z_dim, z_dim, kernel_size=1),
        )

        self.zs = [] #used to store the sampled z for tracking the latent variables generated in different levels

    def forward(self, z, xs=None, mode="random", freeze_level=-1):
        """
        -> mu, log_var, xs = self.encoder(x)
        -> z = reparameterize(mu, torch.exp(0.5 * log_var))
        
        :param z: shape. = (B, z_dim, length) = (batch_size, 1, 64)
        
        -> decoder_out: Info learned by the decoder during the generation process. 
        It starts as an initialized tensor and is updated as the decoding progresses through different layers.
        
        """
        B, D, L = z.shape

        # The init h (hidden state), can be replace with learned param?
        decoder_out = torch.zeros(B, D, L, device=z.device, dtype=z.dtype) #decoder_out.shape=[32, 1, 16]
        kl_losses = []
        if freeze_level != -1 and len(self.zs) == 0 :
            self.zs.append(z)
        for i in range(len(self.decoder_blocks)):
            #! The samples (z) are combined with deterministic feature maps (decoder_out) and passed to the next group
            z_sample = torch.cat([decoder_out, z], dim=1) #dim=1:z_sample.shape=[32,2,64] #***Q:cat-channel or cat-length?
            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))
            if i == len(self.decoder_residual_blocks) - 1: #6
                break
            mu_and_var = self.condition_z[0](decoder_out)
            mu, log_var = self.condition_z[1](mu_and_var).split(1, dim=1)
            
            if xs is not None:
                cond_x_z = torch.cat([xs[i], decoder_out], dim=1) #p(z|x,z)
                #***Q: p(z|x,z) the condition of x should be cat-channel or cat-length
                delta_mu, delta_log_var = self.condition_xz[i](cond_x_z).split(1, dim=1)
                kl_2_loss = kl_2(delta_mu, delta_log_var, mu, log_var)
                kl_losses.append(kl_2_loss)
                mu = mu + delta_mu
                log_var = log_var + delta_log_var
            
            if mode == 'fix' and i < freeze_level:
                if len(self.zs) < freeze_level + 1:
                    z = reparameterize(mu, 0)
                    self.zs.append(z)
                else:
                    z = self.zs[i + 1]
            elif mode == "fix":
                z = reparameterize(mu, 0 if i == 0 else torch.exp(0.5 * log_var))
            else:
                z = reparameterize(mu, torch.exp(0.5 * log_var)) #z:24-32-40-48-56
                
        x_hat_sigmoid = torch.sigmoid(self.recon(decoder_out)) #sigmoid?
        x_hat = self.recon(decoder_out)
        #print(x_hat_sigmoid)
        #print(x_hat)
        
        return x_hat, x_hat_sigmoid, kl_losses

class NVAE(nn.Module):
    def __init__(self, z_dim, embedding_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=1, float_dtype=np.float32, device="cpu")

    def forward(self, x):
        """
        :param x: Tensor. shape = (B, C, L)
        :return:
        """
        mu, log_var, xs = self.encoder(x)
        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var))
        x_hat, x_hat_sigmoid, losses = self.decoder(z, xs)

        #KL(q(z1|x) || p(z1))
        kl_loss = kl(mu, log_var)

        kl_losses_stacked = torch.stack(losses, dim=0)
        enc_hier_kl = losses + [kl_loss]
        stack_enc_hier_kl = torch.stack(enc_hier_kl, dim=0)
        kl_avg_loss = torch.mean(stack_enc_hier_kl, dim=0)
        kl_sum_loss = torch.sum(stack_enc_hier_kl, dim=0)
        
        sigmoid_loss = torch.mean(F.binary_cross_entropy(x_hat_sigmoid, x, reduction='none'), dim=[1, 2])[:, None]
        recon_sigmoid_loss = torch.mean(self.adaptive_loss.lossfun(sigmoid_loss))
        recon_mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        recon_mix_loss = reconstruction_loss = 0.1 * torch.nn.functional.l1_loss(x_hat, x) + \
                                    1.0 - torch.nn.functional.cosine_similarity(x_hat, x).mean() + \
                                    0.1 * torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        #print(f"RECOM- sigmoid loss: {recon_sigmoid_loss}, MSE: {recon_mse_loss}, MIX: {recon_mix_loss} ")

        return x_hat, recon_mse_loss, recon_mix_loss, recon_sigmoid_loss, kl_avg_loss, kl_sum_loss, [kl_loss]+losses

#vae = NVAE(1,64)
#rand_embedding = torch.rand(32,1,64)
#x_hat, recon_mse_loss, recon_mix_loss, recon_sigmoid_loss, kl_avg_loss, kl_sum_loss, losses = vae(rand_embedding)
#print('OUT_decoder:', decoder_output.shape, 'OUT_recon:', recon_loss, 'OUT_avg:', average_loss)

#encoder = Encoder(1)
#rand_embedding = torch.rand(32,1,64)
#mu, var, xs = encoder(rand_embedding)
#print(mu.shape, var.shape, xs[0].shape)