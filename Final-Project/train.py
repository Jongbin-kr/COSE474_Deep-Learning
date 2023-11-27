import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import AutoProcessor, AutoModel, AutoConfig
from transformers import VisionTextDualEncoderModel
from transformers import AutoModelForMultipleChoice

from typing import *
from tqdm import tqdm

from preprocess import JsonToDataset, MultimodalDataset, get_data_loader
# from model import CLIPForMultipleChoice

model_checkpoint = "koclip/koclip-base-pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# paths
paths = {
    ## dataset & image path
    'train_path': '/kovar-vol/kovar/dataset/train.json',
    'test_path' : '/kovar-vol/kovar/dataset/test.json',
    'image_path' : '/kovar-vol/images',
}

def load_processor(model_checkpoint:str):
    #  group together two or more processing objects such as tokenizers (for the text modality), image processors (for vision) and feature extractors (for audio).
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    return processor


def load_model(model_checkpoint:str):
    # model: VisionTextDualEncoderModel
    # vision_model : ClipVisionModel
    # config = AutoConfig.from_pretrained(model_checkpoint)
    # vision_model = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
    # text_model = AutoModel.from_pretrained('klue/roberta-large')

    # dual_encoder_model = CLIPForMultipleChoice(config, vision_model, text_model)
    # model = dual_encoder_model.from_pretrained(model_checkpoint)

    model = AutoModel.from_pretrained(model_checkpoint)
    # model.eval()
    return model

def get_inputs(processor, texts:torch.Tensor, images:torch.Tensor):
    # processor = load_processor(model_checkpoint)
    inputs = processor(
        text = texts,
        images = images,
        return_tensors = "pt",
        padding=True
    )
    return inputs

def inference(model, inputs):
    outputs = model(**inputs)

    logits_per_image, logits_per_text = outputs
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)
    return probs

## Dataset
{"IDX":58390,
 "STORY_ID":272522,
 "TYPE":"Photo",
 "TITLE":"어린이",
 "CLUE1":"N1Z0691",
 "OBS1":"한 어린이가 건물 주차장에서 혼자 발길질을 하고 있었다.",
 "OBS2":"어린이는 사탕을 들고 신나하며 고맙다고 말했다.",
 "GROUNDTRUTH":"주차 경비원이 다가와 어린이에게 딸기맛 사탕을 건네주었다.",
 "PLAUSIBLE":"그 어린이에게 경비 아저씨가 사탕을 주었다.",
 "IMPLAUSIBLE":"그 어린이에게 경비 아저씨가 인사를 했다.",
 "RANDOM":"그는 놀러가자는 친구들의 유혹에도 꿋꿋하게 사법 고시를 준비했다.",
 "COUNT":5,
 "hyp0":"그 어린이에게 경비 아저씨가 사탕을 주었다.",
 "hyp1":"그 어린이에게 경비 아저씨가 인사를 했다.",
 "hyp2":"그는 놀러가자는 친구들의 유혹에도 꿋꿋하게 사법 고시를 준비했다.",
 "label":0}

#utils

if __name__ == "__main__":
    model = load_model(model_checkpoint)


    batch_size = 4
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    num_epochs = 30

    
    get_dataset = JsonToDataset()
    train_set = get_dataset(paths["train_path"])
    test_set = get_dataset(paths["test_path"])


    

    train_set = MultimodalDataset(train_set)
    test_set = MultimodalDataset(test_set)

    train_loader = get_data_loader(train_set, batch_size)
    test_loader = get_data_loader(test_set, batch_size)

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch in pbar:
            optimizer.zero_grad()
            inputs, labels = batch
            # inputs.to(device)

            # Forward pass
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']


            # Override 
            num_choices = input_ids.shape[1] if input_ids is not None else None

            # 4 * 3 * 60 -> 12 * 60
            flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
            flat_attention_mask = attention_mask.view(-1, input_ids.size(-1)) if input_ids is not None else None
            flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None


            result = model(input_ids = flat_input_ids, pixel_values=inputs['pixel_values'], attention_mask=flat_attention_mask, token_type_ids = flat_token_type_ids)
            
            # ex) 4 * 12
            logits_per_image = result['logits_per_image']

            # ex) indices = torch.tensor([[0, 1, 2],[3,4,5],[6,7,8],[9,10,11]])
            indices = torch.arange(logits_per_image.shape[1]).reshape(4,3)
            logits_per_image = torch.gather(input=logits_per_image,dim= 1, index = indices)

            # Error
            loss = nn.CrossEntropyLoss(logits_per_image, labels)
            print(loss)
            # Compute loss

            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            # ground_truth = labels
            print('end')

    

    for batch in train_loader:
        print(batch)
        break
