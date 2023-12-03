import logging
from typing import *
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoModel

from preprocess import JsonToDataset, MultimodalDataset, get_data_loader

# TODO
# - [ ] Multi GPU Train
# - [ ] Connect to Wandb
# - [ ] Save model checkpoint


device = "cuda" if torch.cuda.is_available() else "cpu"

# Use sys.args later
configs={
        'batch_size' : 4,
        'num_epochs':30,
        'logging_steps':10,
        'eval_steps':10,
        'model_checkpoint':"koclip/koclip-base-pt"
    }    
paths = {
    "train_path": "/kovar-vol/kovar/dataset/train.json",
    "test_path": "/kovar-vol/kovar/dataset/test.json",
    "image_path": "/kovar-vol/images"}


@dataclass
class TrainingArguments:
    batch_size:int=1
    num_epochs:int=1
    logging_steps:int=1
    eval_steps:int=1
    model_checkpoint:str='koclip/koclip-base-pt'

@dataclass
class DatasetArguments:
    train_path:str=None
    test_path:str=None
    image_path:str=None



# Set Logger
FORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(level=logging.INFO, format=f"{FORMAT}")
logger = logging.getLogger()

def get_inputs_for_multiple_choice(input_ids, attention_mask, token_type_ids, pixel_values
):
    view_size = input_ids.size(-1)
    flat_input_ids = input_ids.view(-1, view_size) if input_ids is not None else None
    flat_attention_mask = attention_mask.view(-1, view_size) if attention_mask is not None else None
    flat_token_type_ids = token_type_ids.view(-1, view_size) if token_type_ids is not None else None

    return {
        "input_ids": flat_input_ids.to(device),
        "pixel_values": pixel_values.to(device),
        "attention_mask": flat_attention_mask.to(device),
        "token_type_ids": flat_token_type_ids.to(device),
    }


def get_loss_for_multiple_choice(logits_per_image, labels):
    labels = torch.tensor(np.array(labels)).to(device)

    # ex) indices = torch.tensor([[0, 1, 2],[3,4,5],[6,7,8],[9,10,11]])
    indices = torch.arange(logits_per_image.shape[1]).reshape(4, 3).to(device)
    logits_per_image = torch.gather(input=logits_per_image, dim=1, index=indices)

    # Compute the loss and return 
    return loss_func(logits_per_image, labels)


def evaluation(test_loader):
    logger.info("=== Evaluation Loop ====")
    eval_pbar = tqdm(test_loader, total=len(test_loader))
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(eval_pbar):
            inputs, labels = batch
            inputs = get_inputs_for_multiple_choice(**inputs)
            result = model(**inputs)
            loss = get_loss_for_multiple_choice(result['logits_per_image'], labels)
            running_loss += loss
    avg_loss = running_loss / (i + 1)
    logger.info("batch %s eval loss: %s" % (i + 1, avg_loss))
    return avg_loss



if __name__ == "__main__":
    # Separate TrainingArguments to ModelArguments later if you need.
    data_config = DatasetArguments(**paths)
    train_config = TrainingArguments(**configs)

    model = model = AutoModel.from_pretrained(train_config.model_checkpoint)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )
    loss_func = nn.CrossEntropyLoss()

    get_dataset = JsonToDataset()
    train_set = MultimodalDataset(get_dataset(data_config.train_path))
    test_set = MultimodalDataset(get_dataset(data_config.test_path))

    train_loader = get_data_loader(train_set, train_config.batch_size)
    test_loader = get_data_loader(test_set, train_config.batch_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_eval_loss = 1_000_000.0

    logger.info("=== Training Loop ===")
    for epoch in range(train_config.num_epochs):
        log_epoch = epoch +1
        logger.info("EPOCH %d" % log_epoch)    
        train_pbar = tqdm(train_loader, total=len(train_loader))
        running_loss = 0

        for i, batch in enumerate(train_pbar):
            # Zero gradients and running loss for every batch
            optimizer.zero_grad()
            log_batch = i+1        
            
            inputs, labels = batch
            inputs = get_inputs_for_multiple_choice(**inputs)
            
            model.train()
            result = model(**inputs)

            # Compute the loss and its gradients
            loss = get_loss_for_multiple_choice(result['logits_per_image'], labels)
            loss.backward()  # TODO : loss.backward가 뭘하는지 확인

            # Adjust learning weights TODO: 이게 뭔지 확인
            optimizer.step()

