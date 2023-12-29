import os
import torch
import time
import logging
import pickle
import wandb
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from nvae import NVAE
#from nvae_4level import NVAE_L4
from nvae_6level import NVAE_L6
from utils import add_sn, Annealer

#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface
#OPENBLAS_NUM_THREADS=128
#export OMP_NUM_THREADS=1

wandb.login()
wandb.init(
    project='NVAE-3towers',
    config={
        "epochs":100,
        "bs":128,
        "lr":0.01,
        "kl_warmup":'beta', #False, tower, linear, beta, cyclic
        "scheduler":'lambda', #None, cosine, lambda, multi
        "scheduler_rate":0.8,
        "recon_loss":'mix', #mix, mse, sigmoid
        "kl_loss":'enc_loss', #avg, sum, enc_loss
        "eva_freq": 2,
        "beta_annealing_percentage":0.2,
        "beta_initial":0.0,
        "beta_final":0.00001,
        "tower_warmup_initial_weights":[1, 0.5, 0.5, 0.5, 0.125, 0.125],
        "tower_warmup_steps":[4500, 3000, 3000, 3000, 1500, 1500],
        "tower_M_N_decay_step":100000,
        "clipping":True,
        "earlystop":False,
    }
)
config = wandb.config
#wandb.init(mode="disabled")

# WarmupKLLoss, n_stages=3(n_tower)
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
        wandb.log({"M_N": M_N, "stage":stage, "step":step, "original KL_loss":loss})
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
    os.makedirs("validation_plot/nvae_retry_beta_t27", exist_ok=True)
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
        decoded_embeddings, _, _, _, _, _, _ = model(real_embeds)
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
        plot_file = f"validation_plot/nvae_retry_beta_t27/epoch_{ep}.png"
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        wandb.log({"visualization":wandb.Image(plot_file)})
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
    beta_annealing_percentage=config.beta_annealing_percentage,
    beta_initial=config.beta_initial,
    beta_final=config.beta_final,
    tower_warmup_initial_weights=config.tower_warmup_initial_weights,
    tower_warmup_steps=config.tower_warmup_steps,
    tower_M_N_decay_step=config.tower_M_N_decay_step,
    clipping=config.clipping
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('* device:', device)
    print(f"* Warmup:{kl_warmup} / scheduler:{scheduler} / scheduler_rate:{scheduler_rate}")
    print(f"* epochs: {epochs} / bs: {batch_size} / lr: {learning_rate} / recon_loss_fn: {recon_loss_fn} / kl_loss_fn: {kl_loss_fn}")
    if kl_warmup == 'beta':
        print(f"* <Beta annealing> beta_annealing_percentage:{beta_annealing_percentage} / beta_initial:{beta_initial} / beta_final:{beta_final}")
    elif kl_warmup == 'tower':
        print(f"* <Tower Warmup> weight:{tower_warmup_initial_weights}, steps:{tower_warmup_steps}, decay_setp:{tower_M_N_decay_step}")
    model = NVAE_L6(z_dim=1, embedding_dim=64)
    #model.apply(add_sn) #?
    model = model.to(device)
    speaker_embedding_dataset = SpeakerEmbeddingDataset()
    dataloader = DataLoader(dataset=speaker_embedding_dataset,
                            batch_size=batch_size, #num_workers=0/drop_last=True
                            shuffle=True)
    
    ## KL annealing
    total_step = (len(speaker_embedding_dataset) // batch_size)*epochs
    step_of_one_epoch = (len(speaker_embedding_dataset) // batch_size)

    warmup_kl_tower = WarmupKLLoss(init_weights=tower_warmup_initial_weights, #[1., 1. / 2, 1. / 8]
                             steps=tower_warmup_steps, #steps=[4500, 3000, 1500]
                             M_N=batch_size / len(speaker_embedding_dataset),
                             eta_M_N=5e-6, #M_N -> 0.000005
                             M_N_decay_step=tower_M_N_decay_step) #36000

    warmup_kl_linear = LinearIncreaseKLLoss(increase_steps=4000)
    cyclic_annealer = Annealer(total_steps=step_of_one_epoch*2, shape='logistic')

    #! Use the AdaMax optimizer for training with the initial learning rate of 0.01 and with cosine 
    #! learning rate decay
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate) #0.01
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #0.01


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
    patience = 5 #5epoch?
    ES_kl_counter, ES_recon_counter = 0,0

    for epoch in range(epochs):
        kl_losses = list()
        reconstruction_losses = list()
        losses = list()
        model.train()
        for i, embedding in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch}'):
            count_step = i
            optimizer.zero_grad()
            embedding = embedding.to(device)
            embedding = embedding.unsqueeze(1) #***Q:will this have any impact to the data
            embedding_recon, recon_loss_mse, recon_loss_mix, recon_loss_sig, kl_loss_avg, kl_loss_sum, tower_losses = model(embedding.to(device))

            ## Recon_loss Type ##
            if recon_loss_fn == 'mse':
                recon_loss = recon_loss_mse
            elif recon_loss_fn == 'mix':
                recon_loss = recon_loss_mix
            elif recon_loss_fn == 'sigmoid':
                recon_loss = recon_loss_sig

            ## KL_loss Type ##
            if (kl_warmup == False) and (kl_loss_fn == 'avg'):
                kl_loss = kl_loss_avg
                loss = recon_loss + kl_loss

            elif (kl_warmup == False) and (kl_loss_fn == 'none'):
                kl_loss = tower_losses[0]
                loss = recon_loss

            elif (kl_warmup == 'beta') and (kl_loss_fn == 'sum'):
                #if step < beta_annealing_percentage * total_step:
                #    beta = beta_initial + (step / (beta_annealing_percentage * total_step)) * (beta_final - beta_initial)
                #else:
                #    beta = beta_final
                if count_step < 1500:
                    beta = 0.0
                elif 1500 <= count_step:
                    beta = beta_initial + (count_step / step_of_one_epoch) * (beta_final - beta_initial)

                kl_loss = kl_loss_sum
                loss = recon_loss + beta*kl_loss
                print(f'count_step:{count_step}, kl_loss:{kl_loss}, beta:{beta}, beta*kl_loss:{beta*kl_loss}')

            elif (kl_warmup == 'beta') and (kl_loss_fn == 'enc_loss'):
                #if step < beta_annealing_percentage * total_step:
                #    beta = beta_initial + (step / (beta_annealing_percentage * total_step)) * (beta_final - beta_initial)
                #else:
                #    beta = beta_final
                
                if count_step < 3000:
                    beta = 0.0
                elif 3000 <= count_step:
                    beta = beta_initial + (count_step / step_of_one_epoch) * (beta_final - beta_initial)

                kl_loss = tower_losses[0]
                loss = recon_loss + beta*kl_loss
                print(f'count_step:{count_step}, kl_loss:{kl_loss}, beta:{beta}, beta*kl_loss:{beta*kl_loss}')
            
            elif (kl_warmup == 'cyclic') and (kl_loss_fn == 'enc_loss'):
                kl_loss = tower_losses[0]
                cyclic_kl_loss = cyclic_annealer(kl_loss)
                print(f'step:{step}, anneal_step:{cyclic_annealer.total_steps}, cyclic_kl_loss:{cyclic_kl_loss}, kl_loss{kl_loss}')
                loss = recon_loss + cyclic_kl_loss

                #if epoch > 5:
                #    loss = recon_loss + 0.2*kl_loss
                #else:
                #    loss = recon_loss + 0.05*kl_loss
                #52:0.05

                #loss = recon_loss + kl_loss
                #if not torch.isnan(kl_loss) and epoch > 10:
                #    loss = kl_loss * 0.2 + recon_loss
                #else:
                #    loss = kl_loss * 0.1 + recon_loss

                #if epoch > 5:
                #    loss = recon_loss + kl_loss
                #else:
                #    loss = recon_loss + 0.5*kl_loss

            elif (kl_warmup == 'tower'):
                kl_loss = warmup_kl_tower.get_loss(step, tower_losses)
                kl_loss = torch.tensor(kl_loss)
                loss = recon_loss + kl_loss
            
            elif (kl_warmup == 'linear'):
                if kl_loss_fn == 'enc_loss':
                    kl_loss = warmup_kl_linear.get_loss(step, enc_kl_loss)
                    kl_loss = torch.tensor(kl_loss)
                    loss = recon_loss + kl_loss

            #log_str = "\r---- [Epoch %d/%d, Step %d/%d] loss: %.5f (recon_loss: %.5f, kl_loss: %.5f)----" % (
            #    epoch, epochs, i, len(dataloader), loss.item(), recon_loss.item(), kl_loss.item())
            #logging.info(log_str)

            kl_losses.append(kl_loss.item())
            reconstruction_losses.append(recon_loss.item())
            losses.append(loss.item())

            loss.backward()
            # Gradient clipping
            if clipping == True:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            cyclic_annealer.step()

            step += 1

        avg_kl_loss = sum(kl_losses) / len(kl_losses)
        avg_reconstruction_loss = sum(reconstruction_losses) / len(reconstruction_losses)
        aggregated_loss = sum(losses) / len(losses)

        lr_value = scheduler.get_lr()[0]
        current_lr = optimizer.param_groups[0]['lr']

        js_pq, js_qp, js_distance, js_pq_2d, js_qp_2d, js_distance_2d = generate_and_plot_samples(model=model, dataset=speaker_embedding_dataset, n_sample=1000, ep=epoch+1)

        log_str_2 = "\r---- [Epoch %d/%d] loss: %.5f (recon_loss: %.5f, kl_loss: %.5f)----" % (
                epoch, epochs, aggregated_loss, avg_reconstruction_loss, avg_kl_loss)
        print(f" ##: {log_str_2}")
        print(f'<JS n-dim> js_pq:{js_pq.mean()}, js_qp:{js_qp.mean()}, js_distance:{js_distance.mean()}')
        print(f'<JS 2-dim> js_pq:{js_pq_2d.mean()}, js_qp:{js_qp_2d.mean()}, js_distance:{js_distance_2d.mean()}')

        metrics = {
            "epoch":epoch,
            "step":step,
            "KL loss": avg_kl_loss,
            "Reconstrunction loss": avg_reconstruction_loss,
            "Loss": aggregated_loss,
            "Learning Rate": lr_value
        }
        wandb.log({**metrics})

        ## Scheduler: called after every 'epoch' not 'batch' ##
        scheduler.step()
        #lr_value = scheduler.get_lr()[0]
        #current_lr = optimizer.param_groups[0]['lr']
        #print(f" * Learning Rate: {current_lr}, {lr_value}, Step: {step}")
        #wandb.log({"Learning Rate": current_lr})

        #model.eval() #***Q:how to evaluate?
        #if epoch % config.eva_freq == 0:
        #    decoded_embeds, decoded_embeds_2d, real_embeds, real_embeds_2d = generate_and_plot_samples(model=model, dataset=speaker_embedding_dataset, n_sample=1000, ep=epoch+1)
        #    print(decoded_embeds.shape, real_embeds.shape)

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

    torch.save(model.state_dict(), "checkpoint/nvae_retry_beta_t27.pt")
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
