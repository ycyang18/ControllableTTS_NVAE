import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import robust_loss_pytorch
from utils import add_sn, Swish, ChannelSELayer1d, reparameterize, kl, kl_2, combined_recon_loss
from blocks import EncoderResidualBlock, DecoderResidualBlock, ResidualBlock, ConvBlock, UpsamplingConvBlock, DownsamplingConvBlock

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
        self.condition_mean = nn.Sequential(
            Swish(),
            nn.Conv1d(z_dim, z_dim, kernel_size=3, padding=1)
        )
        self.condition_var = nn.Sequential(
            Swish(),
            nn.Conv1d(z_dim, z_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        test = []
        xs = []
        last_x = x
        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x = r(e(x))
            last_x = x # last feature map
            xs.append(x)
        mu = self.condition_mean(last_x)
        log_var = self.condition_var(last_x)
        #mu, log_var = self.condition_x(last_x).split(1, dim=1)
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
        self.condition_z_mean = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
        ])

        # p(z_l | z_(l-1))
        self.condition_z_var = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, z_dim, kernel_size=1)
            ),
        ])

        # p(z_l | x, z_(l-1))
        self.condition_xz_mean = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
        ])
        self.condition_xz_var = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(2),
                nn.Conv1d(2, 1, kernel_size=1),
                Swish(),
                nn.Conv1d(1, 1, kernel_size=1)
            ),
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
        decoder_outputs = []
        if freeze_level != -1 and len(self.zs) == 0 :
            self.zs.append(z)
        for i in range(len(self.decoder_blocks)):
            #! The samples (z) are combined with deterministic feature maps (decoder_out) and passed to the next group
            z_sample = torch.cat([decoder_out, z], dim=1) #dim=1:z_sample.shape=[32,2,64] #***Q:cat-channel or cat-length?
            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))
            decoder_outputs.append(decoder_out)
            if i == len(self.decoder_residual_blocks) - 1: #6
                break
            mu, log_var = self.condition_z_mean[i](decoder_out), self.condition_z_var[i](decoder_out)
            
            if xs is not None:
                cond_x_z = torch.cat([xs[i], decoder_out], dim=1) #p(z|x,z)
                #***Q: p(z|x,z) the condition of x should be cat-channel or cat-length
                delta_mu, delta_log_var = self.condition_xz_mean[i](cond_x_z), self.condition_xz_var[i](cond_x_z)
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
                z = reparameterize(mu, torch.exp(0.5*log_var)) ###0.5
                
        x_hat_sigmoid = torch.sigmoid(self.recon(decoder_out)) #sigmoid?
        x_hat = self.recon(decoder_out)
        
        return x_hat, x_hat_sigmoid, kl_losses, decoder_outputs

class NVAE_L6_mod(nn.Module):
    def __init__(self, z_dim, embedding_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=1, float_dtype=np.float32, device="cpu")

    def forward(self, x):
        #if rand_z is not None: #
        #    x_hat, _, _, _ = self.decoder(z=rand_z) #
        #    return x_hat #
        #else: #
        recon_losses = []
        mu, log_var, xs = self.encoder(x)
        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5*log_var)) ###0.5
        x_hat, x_hat_sigmoid, kl_losses, decoder_outputs = self.decoder(z, xs)

        #KL(q(z1|x) || p(z1))
        first_kl_loss = kl(mu, log_var)
        tower_kl_loss = [first_kl_loss] + kl_losses #[16],24,32,40,48,56
        sum_kl_loss = sum(tower_kl_loss)

        ## recon loss ##
        last_recon_loss = combined_recon_loss(x, x_hat) #64
        for _x, _d in zip(xs, decoder_outputs):
            loss = combined_recon_loss(_x, _d) #24-32-40-48-56
            recon_losses.append(loss)
        tower_recon_loss = recon_losses + [last_recon_loss]
        sum_recon_loss = sum(tower_recon_loss)

        return x_hat, tower_recon_loss, sum_recon_loss, tower_kl_loss, sum_kl_loss

def count_parameters(net):
     return sum(p.numel() for p in net.parameters() if p.requires_grad)

vae = NVAE_L6_mod(1,64)
rand_embedding = torch.rand(32,1,64)
x_hat, tower_recon_loss, sum_recon_loss, tower_kl_loss, sum_kl_loss = vae(rand_embedding)
print(count_parameters(vae))

#Note: 
# - increase the number of the parameter
# - add fully connected layers (GAN: in the end of the res-net, add linear block)
# - parameter size: ~60000
# - fine-tuning with libritts-R (with small learning rate): generate clear samples
