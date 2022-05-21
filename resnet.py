import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet_For_Age(nn.Module):
    def __init__(self, model_name='resnet50', pretrained = True):
        super().__init__()
        self.resnet = timm.create_model(model_name = model_name, pretrained = pretrained) 
        self.num_features = self.resnet.num_features
        self.age_fc = nn.Sequential(
                nn.Linear(in_features = self.num_features, out_features = 96),
                nn.ReLU(),
                nn.Linear(in_features = 96, out_features = 1)
            )

        self.gender_fc = nn.Sequential(
                nn.Linear(in_features = self.num_features, out_features = 1),
            )


    def forward_head(self, x):
        x = self.resnet.global_pool(x)
        age = self.age_fc(x)
        gender = self.gender_fc(x)
        return age, gender

    def forward(self, x):
        x = self.resnet.forward_features(x)
        age, gender = self.forward_head(x)
        return age, gender



class Vit_For_Age(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224', pretrained = True):
        super(Vit_For_Age, self).__init__()

        self.vit = timm.create_model(model_name = model_name, pretrained = pretrained) 
        self.embed_dim = self.vit.embed_dim

        self.age_fc = nn.Sequential(
                nn.Linear(in_features = self.embed_dim, out_features = 96),
                nn.ReLU(),
                nn.Linear(in_features = 96, out_features = 1)
            )

        self.gender_fc = nn.Sequential(
                nn.Linear(in_features = self.embed_dim, out_features = 1),
            )

    def forward(self,x):
         x = self.vit.forward_features(x)
         age = self.age_fc(x)
         gender = self.gender_fc(x)
         return age, gender

