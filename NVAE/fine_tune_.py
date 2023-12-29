import os
import torch
import time
import logging
import pickle
import wandb
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from nvae_6level_mod import NVAE_L6_mod
from utils import add_sn, Annealer

wandb.login()
wandb.init(
    project='NVAE_FineTune',
    config={
        "epochs":500,
        "bs":512,
        "lr":0.0001,
        "kl_warmup":'beta', #False, tower, linear, beta, cyclic
        "scheduler":None, #None, cosine, lambda, multi
        "scheduler_rate":0.0,
        "recon_loss":'sum', #last, sum, avg
        "kl_loss":'sum', #enc_loss, sum, avg
        "other_kl_weight":1,
        "first_kl_weight":1,
        "last_kl_weight":1,
        "recon_weight":1,
        "beta_initial":0.0000001,
        "beta_final":0.0001,
        "beta_stay_zero_step":50,
        #"tower_warmup_initial_weights":[1, 0.5, 0.5, 0.5, 0.125, 0.125],
        #"tower_warmup_steps":[4500, 3000, 3000, 3000, 1500, 1500],
        #"tower_M_N_decay_step":100000,
        "clipping_value":0.0,
        "clipping":False,
        "earlystop":False,
    }
)
config = wandb.config
#wandb.init(mode="disabled")

exp_name = 'finetuned_t43'

def generate_and_plot_samples(model, dataset, n_sample, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f"density_plot/xy_axis/{exp_name}", exist_ok=True)
    os.makedirs(f"density_plot/combined/{exp_name}", exist_ok=True)
    os.makedirs(f"validation_plot/{exp_name}", exist_ok=True)

    real_embeds = list()
    decoded_embeds = list()
    for _ in range(n_sample):
        index = np.random.randint(0, len(dataset))
        sampled_embedding = torch.tensor(dataset[index]).unsqueeze(0)
        real_embeds.append(sampled_embedding)
    real_embeds = torch.stack(real_embeds)
    #print('real_embeds:', real_embeds.shape) [1000,1,64]

    with torch.no_grad():
        real_embeds = real_embeds.to(device) #
        decoded_embeddings, _, _, _, _ = model(real_embeds) #
        #sampled_z = torch.randn(n_sample,1,16).to(device) #
        #decoded_embeddings = model(x=None, rand_z=sampled_z) #
        #print(f'sampled_z:{sampled_z.shape}, decoded_embeddings:{decoded_embeddings.shape}')
        decoded_embeddings, real_embeds = decoded_embeddings.squeeze(1), real_embeds.squeeze(1)
        decoded_embeddings = decoded_embeddings.cpu().numpy()
        real_embeds = real_embeds.cpu().numpy()
        embeddings_as_array = np.vstack((decoded_embeddings, real_embeds))

        reduction = TSNE(n_components=2,
                     learning_rate="auto",
                     init="pca",
                     n_jobs=-1)
        scaler = StandardScaler()
        dimensionality_reduced_embeddings = scaler.fit_transform(
        reduction.fit_transform(X=scaler.fit_transform(embeddings_as_array)))

        # JS distance: N-dim
        p = np.array(decoded_embeddings) #predicted
        q = np.array(real_embeds) #real
        #p_dist = p / p.sum(axis=0, keepdims=True)
        #q_dist = q / q.sum(axis=0, keepdims=True)
        m = (p + q) / 2
        divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
        js_pq = scipy.stats.entropy(p, m)
        js_qp = scipy.stats.entropy(q, m)
        js_distance = np.sqrt(divergence)

        # JS distance: 2-dim
        p_2d = np.array(dimensionality_reduced_embeddings[:n_sample]) #predicted
        q_2d = np.array(dimensionality_reduced_embeddings[n_sample:]) #real
        m_2d = (p_2d + q_2d) / 2
        divergence_2d = (scipy.stats.entropy(p_2d, m_2d) + scipy.stats.entropy(q_2d, m_2d)) / 2
        js_pq_2d = scipy.stats.entropy(p_2d, m_2d)
        js_qp_2d = scipy.stats.entropy(q_2d, m_2d)
        js_distance_2d = np.sqrt(divergence_2d)

        # Scatter plot
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
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
        plot_file_scatter = f"validation_plot/{exp_name}/epoch_{ep}.png"
        plt.savefig(plot_file_scatter, bbox_inches='tight')
        plt.close()

        # density plot
        #fig, axs = plt.subplots(1, 3)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        X_plot_combined = np.linspace(min(p_2d.min(), q_2d.min()), max(p_2d.max(), q_2d.max()), 1000)[:, np.newaxis]
        for i, axis_name in enumerate(['x-axis', 'y-axis']):
            kde_p = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(p_2d[:, i].reshape(-1, 1))
            kde_q = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(q_2d[:, i].reshape(-1, 1))
            log_dens_p = kde_p.score_samples(X_plot_combined)
            log_dens_q = kde_q.score_samples(X_plot_combined)
            axs[0].plot(X_plot_combined[:, 0], np.exp(log_dens_p), lw=1.5, label=f'predicted {axis_name}')
            axs[0].plot(X_plot_combined[:, 0], np.exp(log_dens_q), lw=1.5, linestyle="--", label=f'truth {axis_name}')
        axs[0].legend()
        axs[0].set_title('Kernel Density Estimation')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Density')

        for i, axis_name in enumerate(['x-axis', 'y-axis']):
            X_plot = np.linspace(min(p_2d[:, i].min(), q_2d[:, i].min()), max(p_2d[:, i].max(), q_2d[:, i].max()), 1000)[:, np.newaxis]
            kde_p = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(p_2d[:, i].reshape(-1, 1))
            kde_q = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(q_2d[:, i].reshape(-1, 1))
            log_dens_p = kde_p.score_samples(X_plot)
            log_dens_q = kde_q.score_samples(X_plot)
            axs[i+1].plot(X_plot[:, 0], np.exp(log_dens_p), color="blue", lw=1.5, label=f'predicted {axis_name}')
            axs[i+1].plot(X_plot[:, 0], np.exp(log_dens_q), color="red", lw=1.5, linestyle="--", label=f'truth {axis_name}')
            axs[i+1].legend()
            axs[i+1].set_title(f'Kernel Density Estimation for {axis_name}')
            axs[i+1].set_xlabel('Value')
            axs[i+1].set_ylabel('Density')
        plt.tight_layout()
        plt.legend()
        plot_file_xy_density = f"density_plot/xy_axis/{exp_name}/epoch_{ep}.png"
        plt.savefig(plot_file_xy_density)
        plt.close()
        
        #fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        fig, ax = plt.subplots()
        combined_data = np.vstack([p_2d, q_2d])
        X_plot = np.linspace(combined_data.min(), combined_data.max(), 1000)[:, np.newaxis]
        for i, label, color in zip([0, 1], ["predicted x-axis", "predicted y-axis"], ["navy", "cornflowerblue"]):
            kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(p_2d[:, i].reshape(-1, 1))
            log_dens = kde.score_samples(X_plot)
            ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=1.5, linestyle="-", label=f"{label}")
        for i, label, color in zip([0, 1], ["truth x-axis", "truth y-axis"], ["darkorange", "red"]):
            kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(q_2d[:, i].reshape(-1, 1))
            log_dens = kde.score_samples(X_plot)
            ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=1.5, linestyle="--", label=f"{label}")
        ax.legend(loc="upper left")
        ax.set_title('Kernel Density Estimations')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        plt.legend()
        plot_file_combined_density = f"density_plot/combined/{exp_name}/epoch_{ep}.png"
        plt.savefig(plot_file_combined_density)
        plt.close()
        
        wandb.log({"scatter":wandb.Image(plot_file_scatter), "XY Density":wandb.Image(plot_file_xy_density), "Combined Density":wandb.Image(plot_file_combined_density)})
        return js_pq, js_qp, js_distance, js_pq_2d, js_qp_2d, js_distance_2d

def train(
    epochs=config.epochs, 
    batch_size=config.bs, 
    learning_rate=config.lr, 
    recon_loss_fn=config.recon_loss, 
    kl_loss_fn=config.kl_loss, 
    kl_warmup=config.kl_warmup, 
    scheduler=config.scheduler, 
    scheduler_rate=config.scheduler_rate, 
    earlystop=config.earlystop, 
    #beta_annealing_percentage=config.beta_annealing_percentage,
    other_kl_weight=config.other_kl_weight,
    first_kl_weight=config.first_kl_weight,
    last_kl_weight=config.last_kl_weight,
    recon_weight=config.recon_weight,
    beta_initial=config.beta_initial,
    beta_final=config.beta_final,
    beta_stay_zero_step=config.beta_stay_zero_step,
    #tower_warmup_initial_weights=config.tower_warmup_initial_weights,
    #tower_warmup_steps=config.tower_warmup_steps,
    #tower_M_N_decay_step=config.tower_M_N_decay_step,
    clipping=config.clipping,
    clipping_value=config.clipping_value
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('* device:', device)
    print(f"* Warmup:{kl_warmup} / clipping:{clipping} / clipping_value:{clipping_value} / scheduler:{scheduler} / scheduler_rate:{scheduler_rate} / earlystop:{earlystop}")
    print(f"* epochs: {epochs} / bs: {batch_size} / lr: {learning_rate} / recon_loss_fn: {recon_loss_fn} / kl_loss_fn: {kl_loss_fn}")
    if recon_loss_fn == 'weighted_sum':
        print(f'recon_weight:{recon_weight}')
    if kl_loss_fn == 'weighted_sum':
        print(f'first_kl_weight:{first_kl_weight} / other_kl_weight:{other_kl_weight} / last_kl_weight:{last_kl_weight}')
    if kl_warmup == 'beta':
        print(f"* <Beta annealing> beta_stay_zero_step:{beta_stay_zero_step} / beta_initial:{beta_initial} / beta_final:{beta_final}")
    
    model = NVAE_L6_mod(z_dim=1, embedding_dim=64)
    model.load_state_dict(torch.load('/mount/studenten/arbeitsdaten-studenten1/yangyg/NVAE_64/NVAE/checkpoint/mod_beta_t43.pt'))
    model = model.to(device)
    speaker_embedding_dataset = LibriRDataset()
    dataloader = DataLoader(dataset=speaker_embedding_dataset,
                            batch_size=batch_size, #num_workers=0/drop_last=True
                            shuffle=True)
    
    ## KL annealing
    total_step = (len(speaker_embedding_dataset) // batch_size)*epochs
    step_of_one_epoch = (len(speaker_embedding_dataset) // batch_size)

    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate) #0.01
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #0.01

    if scheduler == 'cosine':
        T_max = 25
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
    patience = 10 #5epoch?
    ES_kl_counter, ES_recon_counter = 0,0

    for epoch in range(epochs):
        kl_losses = list()
        reconstruction_losses = list()
        losses = list()
        real_losses = list()
        model.train()
        for i, embedding in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch}'):
            count_step = i
            optimizer.zero_grad()
            embedding = embedding.to(device)
            embedding = embedding.unsqueeze(1)
            x_hat, tower_recon_loss, sum_recon_loss, tower_kl_loss, sum_kl_loss = model(embedding.to(device))

            # look into tower loss
            item_kl_loss, item_recon_loss = [], []
            for _recon, _kl in zip(tower_recon_loss, tower_kl_loss):
                item_recon_loss.append(_recon.item())
                item_kl_loss.append(_kl.item())
            print('* Recon loss:', item_recon_loss)
            print('* KL loss:', item_kl_loss)

            ## Recon_loss Type ##
            if recon_loss_fn == 'last':
                recon_loss = tower_recon_loss[-1]
                og_recon_loss = tower_recon_loss[-1]
            elif recon_loss_fn == 'sum':
                recon_loss = sum_recon_loss
                og_recon_loss = sum_recon_loss
            elif recon_loss_fn == 'weighted_sum':
                recon_loss = sum(tower_recon_loss[:5])+tower_recon_loss[-1]*recon_weight
                og_recon_loss = sum_recon_loss
            elif recon_loss_fn == 'avg':
                recon_loss = sum_recon_loss.mean()
                og_recon_loss = sum_recon_loss.mean()

            ## KL_loss Type ##
            if (kl_warmup == False) and (kl_loss_fn == 'none'):
                kl_loss = tower_kl_loss[0]
                loss = recon_loss
            
            elif (kl_warmup == 'beta') and (kl_loss_fn == 'weighted_sum'):
                kl_loss = (tower_kl_loss[0]*first_kl_weight + sum(tower_kl_loss[1:4])*other_kl_weight + tower_kl_loss[-1]*last_kl_weight)
                og_kl_loss = sum_kl_loss
                if count_step < beta_stay_zero_step:
                    beta = beta_initial
                elif beta_stay_zero_step <= count_step:
                    beta = beta_initial + (count_step / step_of_one_epoch) * (beta_final - beta_initial)
                loss = recon_loss + beta*kl_loss
                print(f'count_step:{count_step}, kl_loss:{kl_loss}, beta:{beta}, beta*kl_loss:{beta*kl_loss}')


            elif (kl_warmup == 'beta') and (kl_loss_fn == 'sum'):
                #if step < beta_annealing_percentage * total_step:
                #    beta = beta_initial + (step / (beta_annealing_percentage * total_step)) * (beta_final - beta_initial)
                #else:
                #    beta = beta_final
                if count_step < beta_stay_zero_step:
                    beta = beta_initial
                elif beta_stay_zero_step <= count_step:
                    beta = beta_initial + (count_step / step_of_one_epoch) * (beta_final - beta_initial)

                kl_loss = sum_kl_loss
                og_kl_loss = sum_kl_loss
                loss = recon_loss + beta*kl_loss
                print(f'count_step:{count_step}, kl_loss:{kl_loss}, beta:{beta}, beta*kl_loss:{beta*kl_loss}')
        
            elif (kl_warmup == 'beta') and (kl_loss_fn == 'avg'):
                if count_step < beta_stay_zero_step:
                    beta = beta_initial
                elif beta_stay_zero_step <= count_step:
                    beta = beta_initial + (count_step / step_of_one_epoch) * (beta_final - beta_initial)

                kl_loss = sum_kl_loss.mean()
                og_kl_loss = sum_kl_loss.mean()
                loss = recon_loss + beta*kl_loss
                print(f'count_step:{count_step}, kl_loss:{kl_loss}, beta:{beta}, beta*kl_loss:{beta*kl_loss}')

            elif (kl_warmup == 'beta') and (kl_loss_fn == 'enc_loss'):
                if count_step < beta_stay_zero_step:
                    beta = beta_initial
                elif beta_stay_zero_step <= count_step:
                    beta = beta_initial + (count_step / step_of_one_epoch) * (beta_final - beta_initial)
                kl_loss = tower_kl_loss[0]
                og_kl_loss = tower_kl_loss[0]
                loss = recon_loss + beta*kl_loss
            
            elif (kl_warmup == 'cyclic') and (kl_loss_fn == 'enc_loss'):
                kl_loss = tower_kl_loss[0]
                og_kl_loss = tower_kl_loss[0]
                cyclic_kl_loss = cyclic_annealer(kl_loss)
                print(f'step:{step}, total_steps:{cyclic_annealer.total_steps}, cyclic_kl_loss:{cyclic_kl_loss}, kl_loss{kl_loss}')
                loss = recon_loss + cyclic_kl_loss

            step_metrics = {
            #"epoch":epoch,
            #"step":step,
            "KL step loss": og_kl_loss,
            "Recon step loss": og_recon_loss
            #"Loss": aggregated_loss,
            #"OG Loss":og_loss,
            #"Learning Rate": current_lr
            }
            wandb.log({**step_metrics})

            combined_loss_before_weighted = og_kl_loss + og_recon_loss

            kl_losses.append(og_kl_loss.item()) #unweighted
            reconstruction_losses.append(og_recon_loss.item()) #unweighted
            real_losses.append(combined_loss_before_weighted.item()) #unweighted
            losses.append(loss.item()) #weighted*

            loss.backward() #weighted*

            # Gradient clipping
            if clipping == True:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipping_value)

            optimizer.step()
            #cyclic_annealer.step()
            step += 1

        avg_kl_loss = sum(kl_losses) / len(kl_losses) #unweighted
        avg_reconstruction_loss = sum(reconstruction_losses) / len(reconstruction_losses) #unweighted
        og_loss = sum(real_losses) / len(real_losses) #unweighted
        aggregated_loss = sum(losses) / len(losses) #weighted*

        #lr_value = scheduler.get_last_lr()
        current_lr = optimizer.param_groups[0]['lr']

        js_pq, js_qp, js_distance, js_pq_2d, js_qp_2d, js_distance_2d = generate_and_plot_samples(model=model, dataset=speaker_embedding_dataset, n_sample=1000, ep=epoch+1)

        log_str_2 = "\r--- [Epoch %d/%d] loss: %.4f , OG_loss: %.4f (recon_loss: %.5f, kl_loss: %.5f)---" % (
                epoch, epochs, aggregated_loss, og_loss, avg_reconstruction_loss, avg_kl_loss)
        print(f" ##: {log_str_2}")
        print(f'lr: {current_lr}')
        print(f'<JS n-dim> js_pq:{js_pq.mean()}, js_qp:{js_qp.mean()}, js_distance:{js_distance.mean()}')
        print(f'<JS 2-dim> js_pq:{js_pq_2d.mean()}, js_qp:{js_qp_2d.mean()}, js_distance:{js_distance_2d.mean()}')

        metrics = {
            "KL loss": avg_kl_loss,
            "Reconstrunction loss": avg_reconstruction_loss,
            "Loss": aggregated_loss,
            "OG Loss":og_loss,
            "Learning Rate": current_lr
        }
        wandb.log({**metrics})

        ## Scheduler: called after every 'epoch' not 'batch' ##
        if scheduler != 'No!':
            scheduler.step()
        #lr_value = scheduler.get_lr()[0]
        #current_lr = optimizer.param_groups[0]['lr']
        #print(f" * Learning Rate: {current_lr}, {lr_value}, Step: {step}")
        #wandb.log({"Learning Rate": current_lr})

        #model.eval() #***Q:how to evaluate?
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"checkpoint_finetune/{exp_name}_ep{epoch}.pt")

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

    torch.save(model.state_dict(), f"checkpoint_finetune/{exp_name}.pt")
    print(f"## loss: {aggregated_loss} (KL: {avg_kl_loss}, recon: {avg_reconstruction_loss})")
    wandb.finish()

class LibriRDataset(Dataset):
    def __init__(self):
        with open('/mount/studenten/arbeitsdaten-studenten1/yangyg/IMS_Toucan_ImprovedControlMulti/embed_pickle/libriR_embeddings.pickle', 'rb') as file:
            self.embedding_list = pickle.load(file)
    def __getitem__(self, index):
        return self.embedding_list[index]
    def __len__(self):
        return len(self.embedding_list)

if __name__ == '__main__':
    dataset = LibriRDataset()
    print(' ### size of the dataset:', dataset.__len__(), ' / steps:', dataset.__len__()//config.bs, ' ### ')
    train()
