import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch.nn.functional as F
import torch
import torchvision

feature_map = None
avgpool_map = None

def get_features(module, input, output):
    global feature_map
    feature_map = output

def get_avgpool(module, input, output):
    global avgpool_map
    avgpool_map = output

def cosine_distance(vec1, vec2):
    return torch.cosine_similarity(vec1, vec2)

orig_dino_img = Image.open('4th semester/lab1/kentozaurus.png').convert('RGB')
orig_target_img = Image.open('4th semester/lab1/abc.jpg').convert('RGB')

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dino_img = preprocess(orig_dino_img).unsqueeze(0)
target_img = preprocess(orig_target_img).unsqueeze(0)

resnet_weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=resnet_weights, progress=False)

model.layer4.register_forward_hook(get_features)
model.avgpool.register_forward_hook(get_avgpool)
model.eval()

_ = model(dino_img)
dino_features = avgpool_map

_ = model(target_img)
target_features = feature_map
print(feature_map.shape)
distances = cosine_distance(dino_features, target_features)

target_img = target_img.squeeze(0).permute(1,2,0).detach().numpy()
distances_resized = F.interpolate(distances.unsqueeze(0), size=(962, 681), mode='bilinear', align_corners=False)
distances = distances_resized.squeeze(0).permute(1,2,0).detach().numpy()
print(distances.shape)
print(target_img.shape)

plt.imshow(orig_target_img)

plt.imshow(distances, cmap='jet', alpha=0.5)
plt.show()

