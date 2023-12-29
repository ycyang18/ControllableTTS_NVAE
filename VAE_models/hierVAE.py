import os
import torch
import pickle
import wandb
import numpy
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
wandb.login()
wandb.init(
    project='hierVAE-embedding',
    config={
        "epochs":5,
        "batch_size":10,
        "lr":0.01,
        "eva_freq": 100,
    }
)
config = wandb.config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAETower(torch.nn.Module):
    def __init__(self, bottleneck_size, device=DEVICE):
        super().__init__()
        self.encoder = Encoder(bottleneck_size=bottleneck_size)
        self.decoder = Decoder(bottleneck_size=bottleneck_size)
        self.prior_distribution = torch.distributions.Normal(0,1)
        self.bottleneck_size = bottleneck_size
        self.device = device
        self.latent = None

    def forward(self, target_data, noise_scale_during_inference=1.4, z=None):
        # Encoding
        if target_data is not None:
            means, variance = self.encoder(target_data)
            means, variance = means.to(self.device), variance.to(self.device)
            variance = variance.exp()
            z = means + variance*self.prior_distribution.sample(means.shape).to(self.device)
            #self.latent = z.to(self.device)
        else:
            if z is None:
                z = torch.randn(self.bottleneck_size).to(self.device).unsqueeze(0)*noise_scale_during_inference
                #print(' * z (VAETower):', z.shape)
            #self.latent = z.to(self.device)

        # Decoding
        reconstructions_of_targets = self.decoder(z)
        #print(' * reconstructed z (VAETower):', reconstructions_of_targets)
    
        if target_data is not None:
            reconstructions_of_targets = self.decoder(z)
            predicted_distribution = torch.distributions.Normal(means,variance)
            kl_loss = torch.distributions.kl_divergence(predicted_distribution, self.prior_distribution).mean()
            reconstruction_loss = 0.1 * torch.nn.functional.l1_loss(reconstructions_of_targets, target_data) + \
                                    1.0 - torch.nn.functional.cosine_similarity(reconstructions_of_targets, target_data).mean() + \
                                    0.1 * torch.nn.functional.mse_loss(reconstructions_of_targets, target_data, reduction="mean")
            return reconstructions_of_targets, kl_loss, reconstruction_loss
        return reconstructions_of_targets

    def get_latent(self):
        return self.latent

class HierVAE(torch.nn.Module):
    def __init__(self, num_towers=5, bottleneck_sizes=[16,16,16,16,16], device=DEVICE, z=None): #torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.num_towers = num_towers
        self.device = device
        self.towers = torch.nn.ModuleList([
            VAETower(bottleneck_size=bottleneck_sizes[i], device=device) for i in range(num_towers)
        ])
    def forward(self, target_data=None, noise_scale_during_inference=1.4, z=None):
        #print('target_data:', target_data, target_data.shape)
        kl_losses = []
        reconstruction_losses = []
        reconstructions = []
        #if target_data in not None:
        if target_data is not None:
            for tower in self.towers:
                #print(target_data, noise_scale_during_inference, z)
                recon, kl_loss, recon_loss = tower(target_data, noise_scale_during_inference, z)
                recon, kl_loss, recon_loss = recon.to(self.device), kl_loss.to(self.device), recon_loss.to(self.device)
                #print(' * KL:', kl_loss, ' / RECON:', recon_loss)
                reconstructions.append(recon)
                kl_losses.append(kl_loss)
                reconstruction_losses.append(recon_loss)
            #print(len(kl_losses))
            return reconstructions, kl_losses, reconstruction_losses
        else:
            reconstructions_decoding = []
            for tower in self.towers:
                reconstructions_of_targets = tower(target_data, noise_scale_during_inference, z)
                reconstructions_decoding.append(reconstructions_of_targets)
            #print(' * reconstructed_last_tower:', reconstructions_decoding[-1])
            return reconstructions_decoding[-1] #take the z from the last tower (shape:[1,64])

class Encoder(torch.nn.Module):
    def __init__(self, bottleneck_size):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,16),
            torch.nn.Tanh(),
            torch.nn.Linear(16,16),
            torch.nn.Tanh(),
            torch.nn.Linear(16,16),
            torch.nn.Tanh(),
            torch.nn.Linear(16,bottleneck_size),
            torch.nn.Tanh()
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
        compressed = self.nn(target_data_for_compression)
        return self.proj_mean(compressed), self.proj_var(compressed)

class Decoder(torch.nn.Module):
    def __init__(self, bottleneck_size):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16,16),
            torch.nn.Tanh(),
            torch.nn.Linear(16,16),
            torch.nn.Tanh(),
            torch.nn.Linear(16,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64)
        )
    def forward(self, compressed_data_for_decompression):
        decompressed = self.nn(compressed_data_for_decompression)
        return decompressed

def create_eval_visualization(net, dataset, epoch):
    #print(' * net():', net().shape, ' (squeeze):', net().squeeze().shape)
    #print(' * net():', net().squeeze().detach().cpu().numpy())
    os.makedirs("validation/HierVAE", exist_ok=True)
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    real_samples_shown = 500
    fake_samples_shown = 500

    generated_embeds = list()
    for _ in range(fake_samples_shown):
        generated_embeds.append(net().squeeze().detach().cpu().numpy()) #.cpu() -> .to(device) 

    real_embeds = list()
    for _ in range(real_samples_shown):
        index = numpy.random.randint(0, len(dataset))
        #real_embeds.append(dataset[index].numpy())
        real_embeds.append(dataset[index])
    reduction = TSNE(n_components=2,
                     learning_rate="auto",
                     init="pca",
                     n_jobs=-1)
    scaler = StandardScaler()

    embeddings_as_array = numpy.array(generated_embeds + real_embeds)
    dimensionality_reduced_embeddings = scaler.fit_transform(
        reduction.fit_transform(X=scaler.fit_transform(embeddings_as_array)))
    for i, datapoint in enumerate(dimensionality_reduced_embeddings):
        if i == 0:
            axes.scatter(x=datapoint[0],
                         y=datapoint[1],
                         c="b" if i < fake_samples_shown else "g",
                         label="fake",
                         alpha=0.4)
        elif i == fake_samples_shown:
            axes.scatter(x=datapoint[0],
                         y=datapoint[1],
                         c="b" if i < fake_samples_shown else "g",
                         label="real",
                         alpha=0.4)
        else:
            axes.scatter(x=datapoint[0],
                         y=datapoint[1],
                         c="b" if i < fake_samples_shown else "g",
                         alpha=0.4)
    axes.axis('off')
    axes.legend()
    plot_file = f"validation/HierVAE/{epoch}.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    wandb.log({"visualization":wandb.Image(plot_file)})

#-------------------------------training-----------------------------------#

print(' * Device:', DEVICE)

def train(epochs=config.epochs, net=HierVAE(device=DEVICE), batch_size=config.batch_size):
    net = net.to(DEVICE)
    #print(' * net():', net())
    net.train()
    torch.backends.cudnn.benchmark = True
    speaker_embedding_dataset = SpeakerEmbeddingDataset()
    dataloader = DataLoader(dataset=speaker_embedding_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True,
                            drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    #create_eval_visualization(net, speaker_embedding_dataset, 0)
    for epoch in range(epochs):
        kl_losses = []
        reconstruction_losses = []
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad() #reset gradients before computing new gradients
            recon_batch, kl_losses_batch, recon_losses_batch = net(target_data=batch_data.to(DEVICE))
            
            #for kl, recon in zip (kl_losses_batch, recon_losses_batch):
            #    print(' * kl:', kl, ' / recon:', recon)
            #    if not torch.isnan(kl) and epoch > 10:
            #        loss = kl * 0.2 + recon
            #        print(f" * kl_loss: {kl}, recon_loss: {recon}, * loss: {loss}")
            #    else:
            #        loss = recon

            kl_losses.extend(kl_losses_batch)
            reconstruction_losses.extend(recon_losses_batch)
            aggregated_loss = sum(kl_losses_batch) + sum(recon_losses_batch)
            if epoch > 5:
                back_loss = 0.5*sum(kl_losses_batch) + sum(recon_losses_batch)
            else:
                back_loss = 0.1*sum(kl_losses_batch) + sum(recon_losses_batch)
            #tuned_aggregated_loss = sum(kl_losses_batch)*0.2 + sum(recon_losses_batch) #weight down the kl_loss
            #print(' * aggregated_loss:', aggregated_loss, ' / tuned_aggregated_loss:', tuned_aggregated_loss)
            back_loss.backward()
            optimizer.step()
        avg_kl_loss = sum(kl_losses) / len(kl_losses)
        avg_reconstruction_loss = sum(reconstruction_losses) / len(reconstruction_losses)
        metrics = {
            "KL loss": avg_kl_loss,
            "back_loss":back_loss,
            "Reconstrunction loss": avg_reconstruction_loss,
            "Aggregated loss": aggregated_loss
        }
        wandb.log({**metrics})

        print(f"Epoch [{epoch+1}/{epochs}], KL Loss: {avg_kl_loss:.4f}, Reconstruction Loss: {avg_reconstruction_loss:.4f}")
        #if epoch % config.eva_freq == 0: #100
        #    create_eval_visualization(net, speaker_embedding_dataset, epoch + 1)
            #create_eval_visualization_wandb(net, speaker_embedding_dataset, epoch + 1)
    #save_directory = '/mount/studenten/arbeitsdaten-studenten1/yangyg/hier_VAE/Models/Checkpoint'
    #model_filename = 'vae_v2_hier.pt'
    #model_path = os.path.join(save_directory, model_filename)
    #torch.save({"model": net.state_dict()}, f=model_path)

class SpeakerEmbeddingDataset(Dataset):
    def __init__(self):
        with open('/mount/studenten/arbeitsdaten-studenten1/yangyg/hier_VAE/all_embeddings.pickle', 'rb') as file:
            self.embedding_list = pickle.load(file)
            #self.embedding_list = self.embedding_list[:100]
    def __getitem__(self, index):
        return self.embedding_list[index]
    def __len__(self):
        return len(self.embedding_list)

if __name__ == '__main__':
   train()