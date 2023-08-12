# loading model weights, generating features from 
import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchsummary import summary
from torchvision.models import resnet50, Wide_ResNet50_2_Weights
import numpy as np
import torch.nn.functional as F

def load_wide_resnet_50(return_nodes:dict=None, verbose =False, size=(3,224,224)):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', Wide_ResNet50_2_Weights.IMAGENET1K_V1, force_reload=True, verbose=False) #weights=Wide_ResNet50_2_Weights.DEFAULT)
    #torch.hub.load('pytorch/vision:v0.14.1', 'wide_resnet50_2', weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1, force_reload=True, verbose=False)#weights=Wide_ResNet50_2_Weights.DEFAULT
    if not return_nodes is None:
        model = create_feature_extractor(model, return_nodes=return_nodes)
    if torch.cuda.is_available():
        model.cuda()
    if verbose:
        summary(model, size)
        for node in get_graph_node_names(model)[1]:
            if 'conv2' in node or True: 
                print(node)
    return model

if __name__ == '__main__':
    return_nodes = {
    'conv1' : "Level_0", # Feature 0
    "layer1.2.conv3": "Level_1", # Feature 1
    "layer2.3.conv3": "Level_2", # Feature 2
    "layer3.5.conv3": "Level_3", # Feature 3
    "layer4.2.conv3": "Level_4", # Feature 4
    }    
    raw = load_wide_resnet_50(return_nodes=return_nodes, verbose=True)