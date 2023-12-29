import os
import torch
import time
import logging
import pickle
import wandb
import soundfile
import numpy as np
import scipy.io
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.utils import spectral_norm
#from NVAE.nvae import NVAE
#from NVAE.nvae_6level import NVAE_L6
#from NVAE.nvae_6level_mod2 import NVAE_L6_mod
from NVAE.nvae_6level_mod_inf import NVAE_L6_mod

#from NVAE.utils import add_sn 
from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface

norm = False
if norm == True:
    with open('/mount/arbeitsdaten31/studenten1/yangyg/NVAE_64/enlarged_1151017_embeddings.pickle', 'rb') as file:
        embedding_list = pickle.load(file)
    data_array = np.stack(embedding_list)
    mean = data_array.mean(axis=0)
    std = data_array.std(axis=0)

def add_sn(m):
    '''
    Constraint/regulize Lipschitz constant (how much  model's weights change)
    -> the encoder output (latent codes) does not change dramatically as its input changes
    -> the latent codes predicted by the encoder remain bounded
    '''
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        return spectral_norm(m)
    else:
        return m

def inference(n_sample):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    #device = 'cpu'
    #pretrained_vae_model = NVAE(1,64) #Warmup_Test19.pt/Warmup_Test2.pt/nvae_retry_tower_t19
    #pretrained_vae_model = NVAE_L6(1,64) #nvae_retry_beta_t11/t10
    pretrained_vae_model = NVAE_L6_mod(1,64) #nvae_mod_beta_t31/t32/t33
    tts_interface = PortaSpeechInterface()
    check_dict_ = torch.load(os.path.join('NVAE/checkpoint_finetune/', 'finetuned_t40_ep100.pt'), map_location=device)    
    pretrained_vae_model.load_state_dict(check_dict_) #strict=False: allow missing key!
    #pretrained_vae_model.apply(add_sn) #
    pretrained_vae_model.eval()
    pretrained_vae_model.to(device)

    n_dim = 16
    random_latent_vectors = torch.randn(n_sample, 1, n_dim).to(device)
    desired_min_variance = 1
    desired_max_variance = 1
    scaling_factor = (desired_max_variance - desired_min_variance) / 2.0
    random_latent_vectors = random_latent_vectors * scaling_factor + (desired_max_variance + desired_min_variance) / 2.0
    with torch.no_grad():
        reconstructed_embeddings = pretrained_vae_model.decoder(z=random_latent_vectors)[0]

        text = "This is a test sentence"
        for _n, _i in enumerate(reconstructed_embeddings): #_i(z:latent variable are all different)
            utterance_embedding = _i.squeeze()
            if norm == True:
                print('normalized')
                input_embed = (utterance_embedding * std) + mean
            else:
                input_embed = utterance_embedding
            print(input_embed.shape)
            generated_audio = tts_interface(text, input_embed)
            generated_audio = generated_audio.cpu().numpy()
            soundfile.write(f"NVAE/best_finetuned_audio/40_{_n}.wav", generated_audio, 24000)
inference(n_sample=20)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
#checkpoint = torch.load(os.path.join('NVAE/checkpoint/', 'Warmup_Test21.pt'), map_location=device)
#print(checkpoint.keys())
