import os
import torch
import pickle

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
directory_path = '/mount/studenten/arbeitsdaten-studenten1/yangyg/IMS_Toucan_ImprovedControlMulti/embed_pickle/'
file_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.pickle')]

all_embedding = []
for file_path in file_paths:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(len(data))
        all_embedding.extend(data)
with open("/mount/studenten/arbeitsdaten-studenten1/yangyg/NVAE_64/enlarged_embeddings.pickle", 'wb') as f:
    pickle.dump(all_embedding, f)

print('total data size:', len(all_embedding))
