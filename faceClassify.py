from asyncio.windows_events import NULL
from xml.parsers.expat import model
from model import build_model
from datasets import build_transform
from PIL import Image
import cv2
import torch
import os

model_name_dict = {
    "resnet50" : "resnet50",
    "VIT_small" : "vit_small_patch16_224",
    "VIT_base" : "vit_base_patch16_224"
}

model_type_dict = {
    "resnet50" : "resnet",
    "VIT_small" : "transformer",
    "VIT_base" : "transformer"
}

model_path_dict = {
    "resnet50" : "resnet50",
    "VIT_small" : "Vit_small_dropout_0.4",
    "VIT_base" : "Vit_base_dropout_0.4"
}

class FaceClassify:
    def __init__(self, parse):
        model_name = model_name_dict[parse.model]
        model_type = model_type_dict[parse.model]
        model_path = model_path_dict[parse.model]
        saved_model = torch.load(f'{parse.root}/{model_path}/best_age.pth')

        self.device = parse.device
        self.transform = build_transform()
        self.model = build_model(model_name=model_name, model_type=model_type)      
        self.model.load_state_dict(saved_model['model'])
        self.model = self.model.to(self.device)

    def classify(self, img, faces):
        if len(faces) == 0:
            return NULL, NULL

        batch_face = list()
        for (left, top, width, height) in faces:
            face = img[max(0, top):top+height, max(0, left):left+width]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)
            face = self.transform(face)
            face = face.reshape(1, 3, 224, 224)
            batch_face.append(face)
        batch_face = torch.cat(batch_face).to(self.device)

        ages, genders = self.model(batch_face)
        
        ages = [round(age[0].item()) for age in ages]
        genders = [gender[0].item() < 0.5 for gender in genders]

        return ages, genders