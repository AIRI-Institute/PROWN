import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from models import resnet34
from tqdm import tqdm

# This scrath is for similarity matri computation 
# Modify path to folder with models
# Only model weights should be in the foloder

folder_path = './models/resnet34'  # Replace with your folder path
def cosine_distance(model1, model2, average='all'):
    similarities = []
    params1, params2 = [], []
    if average == 'all':
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2, f"Layer mismatch: {name1} != {name2}"
            params1.append(param1.data.view(-1))
            params2.append(param2.data.view(-1))
        params1 = torch.cat(params1)
        params2 = torch.cat(params2)

        cosine_sim = cosine_similarity(params1.unsqueeze(0).to('cpu'), params2.unsqueeze(0).to('cpu')).item()
        return cosine_sim  # converting similarity to distance
    
    elif average == 'layer':
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2, f"Layer mismatch: {name1} != {name2}"
            similarities.append(cosine_similarity(param1.unsqueeze(0).to('cpu').detach(), param2.unsqueeze(0).to('cpu').detach()).item())

        return np.mean(similarities)  # converting similarity to distance
    else:
        print('all or layer average implemented')
        raise ValueError

# Function to calculate cosine similarity
def cosine_similarity_matrix(model1, model2):
    params1, params2 = [], []
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Layer mismatch: {name1} != {name2}"
        params1.append(param1.data.view(-1))
        params2.append(param2.data.view(-1))
    params1 = torch.cat(params1)
    params2 = torch.cat(params2)

    cosine_sim = cosine_similarity(params1.unsqueeze(0).to('cpu'), params2.unsqueeze(0).to('cpu')).item()
    return cosine_sim

# Load all models
model_files = [f for f in os.listdir(folder_path)]
models = []
for ind, file in enumerate(model_files):
    model = resnet34(pretrained=False)
    if file.endswith('model_0.pt'):
        print(ind)
        model.load_state_dict(torch.load(os.path.join(folder_path, file), map_location=torch.device('cpu'))['model'])
    else:
        model.load_state_dict(torch.load(os.path.join(folder_path, file), map_location=torch.device('cpu')))
    models.append(model)

# Calculate similarity matrix
similarity_matrix = np.zeros((len(models), len(models)))
for i, model1 in tqdm(enumerate(models)):
    for j, model2 in tqdm(enumerate(models)):
        if j <= i:
            continue
        else:
            similarity_matrix[i, j] = cosine_distance(model1, model2, average='layer')

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=False, cmap='viridis')
plt.title('Model Similarity Matrix')

# Save the heatmap
plt.savefig('./model_similarity_heatmap.png')
