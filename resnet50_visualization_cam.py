import cv2
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn.functional as F
import ast
import os

resnet_weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=resnet_weights)
model.eval()

feature_map = None

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imagenet1000_clsid_to_human.txt')

with open(file_path) as imagenet_classes_file:
        imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def get_hook(module, input, output):
    global feature_map
    feature_map = output

def get_features_map(model, input):
    global feature_map
    hook = model.layer4.register_forward_hook(get_hook)
    model(input)
    hook.remove()
    return feature_map

def get_fc_weights(model):
    return model.fc.weight.data.numpy()

def create_heatmap(frame, model, weights):
    preprocess = resnet_weights.transforms()
    frame_tensor = preprocess(torch.from_numpy(frame / 255.0).float().unsqueeze(0).permute(0, 3, 1, 2))
    # print(f"frame_tensor: {frame_tensor.shape}")

    preds = model(frame_tensor) 
    # print(f"preds: {preds.shape}")
    class_index = torch.argmax(preds, dim=-1).item()

    feature_maps = get_features_map(model, frame_tensor).squeeze().detach().numpy()
    # print(f"feature maps: {feature_maps.shape}")

    amp_layer_weights = weights[class_index]
    # print(f"amp_layer_weights: {amp_layer_weights.shape}")

    feature_maps_tensor = torch.from_numpy(feature_maps).unsqueeze(0)
    mat_for_mult = F.interpolate(feature_maps_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    mat_for_mult = mat_for_mult.squeeze(0).permute(1, 2, 0).numpy()
    # print(f"mat_for_mult: {mat_for_mult.shape}")

    result = np.dot(mat_for_mult.reshape((224*224, 2048)), amp_layer_weights).reshape(224, 224)
    result = np.maximum(result, 0)
    result = result / np.max(result)

    # print(f"result: {result.shape}")
    print(f"prediction: {imagenet_classes_dict[class_index]}")

    return result, class_index

def print_heatmap(frame, model, weights):
    heatmap, class_index = create_heatmap(frame, model, weights)

    resized_heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    resized_heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)

    frame = (frame * 255).astype(np.uint8)
    result = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)
    result = cv2.putText(result, org=(50, 50), text=imagenet_classes_dict[class_index], color=(0,0,0), fontFace=cv2.LINE_AA,\
            fontScale=1, thickness=3)
    return result

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

fc_weights = get_fc_weights(model)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame.")
        break

    frame_rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))

    frame_with_heatmap = print_heatmap(frame_rgb, model, fc_weights)
    
    # print(frame_rgb.shape, frame_with_heatmap.shape)
    output_frame = cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (640, 480))
    output_heatmap = cv2.resize(frame_with_heatmap, (640, 480))

    cv2.imshow('ResNet50', np.concatenate((output_frame, output_heatmap), axis=1))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()