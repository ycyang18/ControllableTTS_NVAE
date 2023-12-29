import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import robust_loss_pytorch
from NVAE.nvae import NVAE
from Utility.storage_config import MODELS_DIR
from InferenceInterfaces.PortaSpeechInterface import PortaSpeechInterface


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 1
latent_dim = 16
tts_interface = PortaSpeechInterface()
model = NVAE(z_dim=z_dim, embedding_dim=64)
model.to(device)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "Checkpoint", "Warmup_Test19.pt"), map_location=device), strict=False)

z = torch.randn((1, 1, latent_dim)).to(device)
freeze_level = 1
m =freeze_level 

level = [16, 32, 40, 48, 56, 64]
#control_level = list(range(0, level[freeze_level-1]))

s = list(range(0, level[freeze_level])) #[5,10,15,20,25,30,35,40,45] 
alpha = 0.1

zs = model.decoder.zs
printed_once = False
text = "hello, how are you? "

while True: #real-time interactive control
    key = cv2.waitKey(200) #capture keyboard input
    with torch.no_grad():
        # mode='fix': Fix the latent variables at certain levels, controlled by freeze_level.
        gen_embeds, x_hat_sigmoid, kl_losses = model.decoder(z, mode='fix', freeze_level=freeze_level)
        if not printed_once:
            print(zs[-1])
            printed_once = True
        #print(zs[-1][0,0,s[m]])
        for gen_embed in gen_embeds:
            generated_audio = tts_interface(text, gen_embed)
            print(generated_audio.shape)
            gen_embed = gen_embed.cpu().numpy()
            gen_embed = cv2.resize(gen_embed, (500, 50))

            cv2.putText(gen_embed, str(s[m]) + ',' + str(zs[-1][0, 0, s[m]].item()), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 255, 255), thickness=1)
            cv2.imshow("hidden:", gen_embed)
            
    if key == ord('w'):
        zs[-1][:, :, s[m]] += alpha
        print(f"level: {zs[-1].shape}/ dim: {m} / value: {zs[-1][:, :, s[m]]}")
        # zs[-1] = torch.randn(1, 1, 24)
    elif key == ord('s'):
        zs[-1][:, :, s[m]] -= alpha
        print(f"level: {zs[-1].shape}/ dim: {m} / value: {zs[-1][:, :, s[m]]}")
    elif key == ord('a'):
        m = (m - 1) % len(s)
        print(f"level: {zs[-1].shape}/ dim: {m} / value: {zs[-1][:, :, s[m]]}")
    elif key == ord('d'):
        m = (m + 1) % len(s)
        print(f"level: {zs[-1].shape}/ dim: {m} / value: {zs[-1][:, :, s[m]]}")
    elif key == ord('q'):
        exit(0)