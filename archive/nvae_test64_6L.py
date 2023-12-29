import os
import torch
import time
import logging
import pickle
import wandb
import numpy as np
import torch.nn as nn
import scipy.stats
import matplotlib.pyplot as plt
import robust_loss_pytorch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from scipy.stats import entropy
from scipy.spatial import distance
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from utils import add_sn, Swish, ChannelSELayer1d, reparameterize, kl, kl_2

wandb.login()
wandb.init(
    project='NVAE-64dim-4tower',
    config={
        "epochs":101,
        "bs":64,
        "lr":0.01,
        "warm":True,
        "warmup_type":'beta', #tower, beta, linear
        "earlystop":False,
        "scheduler":'lambda', #None, cosine, lambda, multi
        "scheduler_rate":0.85,
        "recon_loss":'mix', #mix, mse, sigmoid
        "kl_loss_type":'enc', #enc, sum
        "eva_freq": 2
    }
)
config = wandb.config
#wandb.init(mode="disabled")

# Encoder
class EncoderResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm1d(dim), Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim), Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            #ChannelSELayer1d(dim) #?
        )
        
    def forward(self, x):
        return x + 0.1 * self.seq(x)

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=5, padding=0),
            nn.Conv1d(out_channel, out_channel, kernel_size=5, padding=0),
            nn.BatchNorm1d(out_channel),
            Swish(),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            Swish()
        )
    def forward(self, x):
        return self._seq(x)


class EncoderBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        modules.append(DownsamplingBlock(channels[0], channels[1]))
        self.modules_list = nn.ModuleList(modules)
        
    def forward(self, x):
        for i, module in enumerate(self.modules_list):
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
            nn.Conv1d(z_dim, z_dim * 2, kernel_size=1)
        )

    def forward(self, x):
        xs = []
        last_x = x
        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x = r(e(x))
            last_x = x
            xs.append(x)
        mu, log_var = self.condition_x(last_x).chunk(2, dim=1)

        return mu, log_var, xs[:-1][::-1]

# Decoder
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim), 
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            #SELayer(dim)
        )
    def forward(self, x):
        return x + 0.1 * self._seq(x)

class DecoderResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.BatchNorm1d(dim), #?
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim), Swish(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(dim), Swish(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim), 
            #ChannelSELayer1d(dim) #?
            
        )
    def forward(self, x):
        return x + 0.1 * self._seq(x)


class UpsamplingConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=1),
            nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=1),
            nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(out_channel),
            Swish(),
            nn.Conv1d(out_channel, 1, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(1),
            Swish()
        )
    def forward(self, x):
        return self._seq(x)


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
            #print(f'DncoderBlock {i+1} output shape:', x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
            DecoderBlock([z_dim*2, z_dim*2]),
        ])
        self.decoder_residual_blocks = nn.ModuleList([
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
            DecoderResidualBlock(z_dim),
        ])

        # p(z_l | z_(l-1))
        self.condition_z = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, 2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim),
                Swish(),
                nn.Conv1d(z_dim, 2, kernel_size=1)
            ),
        ])

        # p(z_l | x, z_(l-1))
        self.condition_xz = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim*2),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1),
                Swish(),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim*2),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1),
                Swish(),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim*2),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1),
                Swish(),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim*2),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1),
                Swish(),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim*2),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1),
                Swish(),
                nn.Conv1d(z_dim*2, z_dim*2, kernel_size=1)
            ),
        ])

        self.recon = nn.Sequential(
            ResidualBlock(z_dim),
            nn.Conv1d(z_dim, z_dim, kernel_size=1),
        )

        self.zs = []

    def forward(self, z, xs=None, mode="random", freeze_level=-1):

        B, D, L = z.shape
        decoder_out = torch.zeros(B, D, L, device=z.device, dtype=z.dtype)

        kl_losses = []
        if freeze_level != -1 and len(self.zs) == 0 :
            self.zs.append(z)

        for i in range(len(self.decoder_residual_blocks)):
            z_sample = torch.cat([decoder_out, z], dim=1)
            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))

            if i == len(self.decoder_residual_blocks) - 1:
                break
            mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1)

            if xs is not None:
                delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs[i], decoder_out], dim=1)).chunk(2, dim=1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            if mode == "fix" and i < freeze_level:
                if len(self.zs) < freeze_level + 1:
                    z = reparameterize(mu, 0)
                    self.zs.append(z)
                else:
                    z = self.zs[i + 1]
            elif mode == "fix":
                z = reparameterize(mu, 0 if i == 0 else torch.exp(0.5 * log_var))
            else:
                z = reparameterize(mu, torch.exp(0.5 * log_var))
        
        x_hat = self.recon(decoder_out)
        x_hat_sigmoid = torch.sigmoid(self.recon(decoder_out))

        return x_hat, x_hat_sigmoid, kl_losses



class NVAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=1, float_dtype=np.float32, device="cpu")

    def forward(self, x):
        mu, log_var, xs = self.encoder(x)
        z = reparameterize(mu, torch.exp(0.5 * log_var))
        x_hat, x_hat_sigmoid, losses = self.decoder(z, xs)
        # Treat p(x|z) as discretized_mix_logistic distribution cost so much, this is an alternative way
        # witch combine multi distribution.
        recon_sig_loss = torch.mean(self.adaptive_loss.lossfun(
            torch.mean(F.binary_cross_entropy(x_hat_sigmoid, x, reduction='none'), dim=[1, 2])[:, None]))
        recon_mix_loss = reconstruction_loss = 0.1 * torch.nn.functional.l1_loss(x_hat, x) + \
                            1.0 - torch.nn.functional.cosine_similarity(x_hat, x).mean() + \
                            0.1 * torch.nn.functional.mse_loss(x_hat, x, reduction="mean")

        kl_loss = kl(mu, log_var)
        tower_kl_loss = [kl_loss] + losses
        sum_kl_loss = sum(tower_kl_loss)
        return x_hat_sigmoid, x_hat, recon_sig_loss, recon_mix_loss, sum_kl_loss, tower_kl_loss

## training ##
class WarmupKLLoss:
    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5, #0.00001
                 M_N_decay_step=3000):
        """
        :param init_weights: initial weights of kl_loss at every level(hierarchy)
        :param steps: steps required for each level to reach w=1.0
        :param M_N: initial M_N (scaling factor applied to the loss, gradually decreases during the decay phase)
        :param eta_M_N: minimum M_N
        :param M_N_decay_step:
        """
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while True:
            if self.stage > len(self.steps) - 1:
                break
            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1
        return self.stage

    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)
        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.) #starting from w=0 to w=1
            elif i < stage:
                w = 1.
            else: #i>stage
                w = self.init_weights[i]
            
            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                # M_N decay:
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(self.M_N - self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
            #print(' ## Ready for M_N (decay):', M_N)
        else:
            M_N = self.M_N
        wandb.log({"M_N": M_N, "stage":stage, "step":step})
        return M_N * loss

class LinearIncreaseKLLoss:
    def __init__(self, increase_steps):
        """
        :param increase_steps: Number of steps for linearly increasing KL loss.
        """
        self.increase_steps = increase_steps

    def get_loss(self, step, loss):
        if step < self.increase_steps:
            weight = step / self.increase_steps
        else:
            weight = 1.
        return weight * loss

def generate_and_plot_samples(model, dataset, n_sample, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("validation_plot/64d_6L_Beta_enc_t9", exist_ok=True)
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    real_embeds = list()
    decoded_embeds = list()
    for _ in range(n_sample):
        index = np.random.randint(0, len(dataset))
        sampled_embedding = torch.tensor(dataset[index]).unsqueeze(0)
        real_embeds.append(sampled_embedding)
    real_embeds = torch.stack(real_embeds)

    with torch.no_grad():
        real_embeds = real_embeds.to(device)
        decoded_embeddings, _, _, _, _, _ = model(real_embeds)
        decoded_embeddings, real_embeds = decoded_embeddings.squeeze(1), real_embeds.squeeze(1)
        decoded_embeddings = decoded_embeddings.cpu().numpy()
        real_embeds = real_embeds.cpu().numpy()
        embeddings_as_array = np.vstack((decoded_embeddings, real_embeds))

        #calculate JS distance:
        p = np.array(decoded_embeddings) #predicted
        q = np.array(real_embeds) #real
        p_dist = p / p.sum(axis=0, keepdims=True)
        q_dist = q / q.sum(axis=0, keepdims=True)
        m = (p + q) / 2
        divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
        js_pq = scipy.stats.entropy(p, m)
        js_qp = scipy.stats.entropy(q, m)
        js_distance = np.sqrt(divergence)

        reduction = TSNE(n_components=2,
                     learning_rate="auto",
                     init="pca",
                     n_jobs=-1)
        scaler = StandardScaler()
        dimensionality_reduced_embeddings = scaler.fit_transform(
        reduction.fit_transform(X=scaler.fit_transform(embeddings_as_array)))
        for i, datapoint in enumerate(dimensionality_reduced_embeddings):
            if i == 0:
                axes.scatter(x=datapoint[0],
                            y=datapoint[1],
                            c="b" if i < n_sample else "g",
                            label="fake",
                            alpha=0.4)
            elif i == n_sample:
                axes.scatter(x=datapoint[0],
                            y=datapoint[1],
                            c="b" if i < n_sample else "g",
                            label="real",
                            alpha=0.4)
            else:
                axes.scatter(x=datapoint[0],
                            y=datapoint[1],
                            c="b" if i < n_sample else "g",
                            alpha=0.4)
        axes.axis('off')
        axes.legend()
        plot_file = f"validation_plot/64d_6L_Beta_enc_t9/epoch_{ep}.png"
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        wandb.log({"visualization":wandb.Image(plot_file)})
        return js_pq, js_qp, js_distance
        #return embeddings_as_array[:n_sample], dimensionality_reduced_embeddings[:n_sample], embeddings_as_array[n_sample:], dimensionality_reduced_embeddings[n_sample:] #decoded, real

def train(epochs=config.epochs, batch_size=config.bs, learning_rate=config.lr, recon_loss_fn=config.recon_loss, kl_loss_type=config.kl_loss_type, kl_warmup=config.warm, scheduler=config.scheduler, scheduler_rate=config.scheduler_rate, earlystop=config.earlystop, warmup_type=config.warmup_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('* device:', device)
    print(f"* Warmup:{kl_warmup} / warmup_type:{warmup_type} / scheduler:{scheduler} / scheduler_rate:{scheduler_rate}")
    print(f"* epochs: {epochs} / bs: {batch_size} / lr: {learning_rate} / recon_loss_fn: {recon_loss_fn} / kl_loss_type:{kl_loss_type}")
    model = NVAE(z_dim=1)

    # apply Spectral Normalization!
    #model.apply(add_sn)
    model = model.to(device)
    speaker_embedding_dataset = SpeakerEmbeddingDataset()
    dataloader = DataLoader(dataset=speaker_embedding_dataset,
                            batch_size=batch_size, #num_workers=0/drop_last=True
                            shuffle=True)

    warmup_kl_tower = WarmupKLLoss(init_weights=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                             steps=[2000, 2000, 2000, 2000, 2000, 2000],
                             M_N=batch_size / len(speaker_embedding_dataset),
                             eta_M_N=5e-6, #M_N -> 0.000005
                             M_N_decay_step=50000) #36000

    warmup_kl_linear = LinearIncreaseKLLoss(increase_steps=4000)

    #! Use the AdaMax optimizer for training with the initial learning rate of 0.01 and with cosine 
    #! learning rate decay
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    ## KL annealing
    total_step = (len(speaker_embedding_dataset) // batch_size)*epochs
    beta_annealing_percentage = 0.2 #first 20% of the training
    beta_initial = 0.0
    beta_final = 0.9

    if scheduler == 'cosine':
        T_max = epochs // 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-4) #eta_min: min lr. Default: 0.
    elif scheduler == 'lambda':
        lambda1 = lambda epoch: scheduler_rate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif scheduler == 'multi':
        lmbda = lambda epoch: scheduler_rate ** epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif scheduler == None:
        scheduler = "No!"
    step = 0

    ## Earlystopping setup ##
    best_kl_loss, best_recon_loss = float('inf'), float('inf')
    patience = 5
    ES_kl_counter, ES_recon_counter = 0,0

    for epoch in range(epochs):
        model.train()
        kl_losses, reconstruction_losses, losses = list(), list(), list()
        for i, embedding in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch}'):
            embedding = embedding.to(device)
            embedding = embedding.unsqueeze(1)
            x_hat_sigmoid, x_hat, recon_sig_loss, recon_mix_loss, sum_kl_loss, tower_kl_loss = model(embedding)

            ## Recon_loss Type ##
            if recon_loss_fn == 'sigmoid':
                recon_loss = recon_sig_loss
            elif recon_loss_fn == 'mix':
                recon_loss = recon_mix_loss

            ## Warmup & KL Type ##
            if step < beta_annealing_percentage * total_step:
                beta = beta_initial + (step / (beta_annealing_percentage * total_step)) * (beta_final - beta_initial)
            else:
                beta = beta_final

            if (kl_warmup == True) and (warmup_type == 'tower'):
                kl_loss = warmup_kl_tower.get_loss(step, tower_kl_loss)
                kl_loss = torch.tensor(kl_loss)
                loss = recon_loss + kl_loss
            
            elif (kl_warmup == True) and (warmup_type == 'linear'):
                kl_loss = warmup_kl_linear.get_loss(step, sum_kl_loss)
                kl_loss = torch.tensor(kl_loss)
                loss = recon_loss + kl_loss
            
            elif (kl_warmup == True) and (warmup_type == 'beta') and (kl_loss_type == 'enc'):
                kl_loss = tower_kl_loss[0]
                loss = recon_loss + beta*kl_loss
            
            elif (kl_warmup == True) and (warmup_type == 'beta') and (kl_loss_type == 'sum'):
                kl_loss = sum_kl_loss
                loss = recon_loss + beta*kl_loss
            
            elif (kl_warmup == False) and (kl_loss_type == 'sum'):
                look_into_tower_loss = list()
                kl_loss = sum_kl_loss
                for _loss in tower_kl_loss:
                    look_into_tower_loss.append(_loss.item())
                print('sum kl loss:', look_into_tower_loss)
                #kl_loss = 0.005 * (tower_kl_loss[0] + 1 / 2 * tower_kl_loss[1] + 1 / 8 * tower_kl_loss[2])
                kl_loss = torch.tensor(kl_loss)
                loss = recon_loss #+ kl_loss

            log_str = "\r---- [Epoch %d/%d, Step %d/%d] loss: %.5f (recon_loss: %.5f, kl_loss: %.5f)----" % (
                epoch, epochs, i, len(dataloader), loss.item(), recon_loss.item(), kl_loss.item())
            #logging.info(log_str)

            kl_losses.append(kl_loss.item())
            reconstruction_losses.append(recon_loss.item())
            losses.append(loss.item())

            optimizer.zero_grad() #
            loss.backward()
            optimizer.step()

            step += 1

        avg_kl_loss = sum(kl_losses) / len(kl_losses)
        avg_reconstruction_loss = sum(reconstruction_losses) / len(reconstruction_losses)
        aggregated_loss = sum(losses) / len(losses)

        lr_value = scheduler.get_lr()[0]
        current_lr = optimizer.param_groups[0]['lr']

        js_pq, js_qp, js_distance = generate_and_plot_samples(model=model, dataset=speaker_embedding_dataset, n_sample=1000, ep=epoch+1)
        log_str_2 = "\r---- [Epoch %d/%d] loss: %.5f (recon_loss: %.5f, kl_loss: %.5f)----" % (
                epoch, epochs, aggregated_loss, avg_reconstruction_loss, avg_kl_loss)
        print(f" ##: {log_str_2}")
        print(f'js_pq:{js_pq.mean()}, js_qp:{js_qp.mean()}, js_distance:{js_distance.mean()}')


        metrics = {
            "epoch":epoch,
            "step":step,
            "KL loss": avg_kl_loss,
            "Reconstrunction loss": avg_reconstruction_loss,
            "Loss": aggregated_loss,
            "Learning Rate": lr_value,
            "Beta":beta,
        }
        wandb.log({**metrics})

        scheduler.step() #scheduler: called after every 'epoch' not 'batch'

        #model.eval()
        #js_pq, js_qp, js_distance = generate_and_plot_samples(model=model, dataset=speaker_embedding_dataset, n_sample=1000, ep=epoch+1)
        #print(f'js_pq:{js_pq.mean()}, js_qp:{js_qp.mean()}, js_distance:{js_distance.mean()}')

        ## Apply Earlystopping
        if earlystop == True:
            if (best_kl_loss < avg_kl_loss) and (best_recon_loss < avg_reconstruction_loss):
                ES_kl_counter += 1
                ES_recon_counter += 1
            else:
                best_kl_loss = avg_kl_loss
                best_recon_loss = avg_reconstruction_loss
                ES_kl_counter, ES_recon_counter = 0,0
            if ES_kl_counter >= patience:
                print(f"## Early stop at epoch: {epoch} | loss: {aggregated_loss} (KL: {avg_kl_loss}, recon: {avg_reconstruction_loss})")
                break

    torch.save(model.state_dict(), "checkpoint/64d_6L_Beta_enc_t9.pt")
    print(f"## loss: {aggregated_loss} (KL: {avg_kl_loss}, recon: {avg_reconstruction_loss})")
    wandb.finish()

class SpeakerEmbeddingDataset(Dataset):
    def __init__(self):
        with open('/mount/studenten/arbeitsdaten-studenten1/yangyg/hier_VAE/all_embeddings.pickle', 'rb') as file:
            self.embedding_list = pickle.load(file)
    def __getitem__(self, index):
        return self.embedding_list[index]
    def __len__(self):
        return len(self.embedding_list)

if __name__ == '__main__':
    dataset = SpeakerEmbeddingDataset()
    print(' ### size of the dataset:', dataset.__len__(), ' / steps:', dataset.__len__()//config.bs, ' ### ')
    train()
