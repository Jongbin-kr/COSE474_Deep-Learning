import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from datasets import Dataset


model_checkpoint = "koclip/koclip-base-pt"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)

# paths
paths = {
    ## dataset & image path
    "train_path": "/kovar-vol/kovar/dataset/train.json",
    "test_path": "/kovar-vol/kovar/dataset/test.json",
    "image_path": "/kovar-vol/images",
}



class JsonToDataset:
    '''
    Convert json files of train/test set to dataset object of hugginface datasets
    - add 'image_path'and 'input_prompt' columns to original DataFrame
    - 'image_path' : abs_path for each images
    - 'input_prompt' : list of str, ['{obs1}[sep]{hyp0}[sep]{obs2}', '{obs1}[sep]{hyp1}[sep]{obs2}', ...]
    '''
    def __init__(self):
        self.tokenizer = processor.tokenizer
        # # Not used now
        # self.max_seq_length = 0

    def __call__(self, json_path: str) -> Dataset:
        df = pd.read_json(json_path, lines=True)
        df["image_path"] = df["CLUE1"].map(self._get_image_paths)
        df["input_prompt"] = df.apply(lambda row: self._format_texts(row), axis=1)
        dataset = Dataset.from_pandas(df)
        return dataset

    def _get_image_paths(self, image_id: str) -> str:
        path = os.path.join(paths["image_path"], image_id[:3], f"{image_id}.jpg")
        return path

    def _format_texts(self, row: pd.DataFrame) -> List[str]:
        sep_token = self.tokenizer.sep_token
        obs1 = row["OBS1"]
        obs2 = row["OBS2"]
        hyps = row.loc["hyp0":"hyp2"]
        text_list = list()
        for hyp in hyps.values:
            prompt_format = sep_token.join([obs1, hyp, obs2])
            # self.max_seq_length = max(len(prompt_format), self.max_seq_length)
            text_list.append(prompt_format)
        return text_list

    # def get_max_seq_length(self):
    #     return self.max_seq_length


class MultimodalDataset:
    '''
    Dataset for KoVAR task based on koCLIP
    - return image, text, label when __getitem__ is called
    - image : an image (PIL.Image.Image)
    - text : List of str, several choices of hypothesis and observations
    - label : the index of sentence that contains a plausible hypothesis in the 'text'.
    '''
    def __init__(self, dataset: Dataset):
        self.tokenizer = processor.tokenizer

        self.image_paths = dataset["image_path"]
        self.texts = dataset["input_prompt"]
        self.labels = dataset["label"]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple:
        image = self._load_image(idx)
        text = self._load_texts(idx)
        label = self._load_label(idx)
        return image, text, label

    def _load_image(self, idx: int) -> Image.Image:
        path = self.image_paths[idx]
        image = Image.open(path)
        return image

    def _load_texts(self, idx: int) -> List[str]:
        return self.texts[idx]

    def _load_label(self, idx: int) -> int:
        return self.labels[idx]


def collate_fn(examples: List[tuple]):
    """
    example[0] = images
    example[1] = texts
    example[2] = labels
    """
    examples = list(filter(lambda x: x is not None, examples))
    

    # make labels
    num_hyp = 3  # temperally fixed
    labels_idx = [example[2] for example in examples]
    labels = np.zeros((len(examples), num_hyp))
    for i, label in enumerate(labels_idx):
        labels[i][label] = 1
    
    # make list of dicts
    inputs = []   # inputs: List[dict]
    lengths = []  # lengths: List[int]
    for example in examples:
        input = processor(images=example[0], text=example[1], return_tensors='pt', padding=True )
        lengths.append(input["input_ids"].shape[1])  # 3 X N
        inputs.append(input)
    
    # dynamic padding in batch
    max_length = max(lengths)
    for idx, input in enumerate(inputs):
        length = lengths[idx]
        num_pad = max_length - length

        pad_token_id = processor.tokenizer.pad_token_id

        input['input_ids'] = np.pad(input['input_ids'], ((0, 0), (0, num_pad)), 'constant', constant_values=pad_token_id)
        input['token_type_ids'] = np.pad(input['token_type_ids'], ((0, 0), (0, num_pad)), 'constant', constant_values=0)
        input['attention_mask'] = np.pad(input['attention_mask'], ((0, 0), (0, num_pad)), 'constant', constant_values=0)
    
    # merge to 1 dict
    padded_inputs = dict()
    keys = ['pixel_values','input_ids', 'attention_mask', 'token_type_ids']
    for key in keys:
        if key == 'pixel_values':
            padded_inputs[key] = torch.stack([input[key][0] for input in inputs])
        else:
            padded_inputs[key] = torch.tensor([input[key] for input in inputs]).reshape(-1, max_length)
            

    return padded_inputs, labels


def get_data_loader(dataset, batch_size):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return data_loader


if __name__ == "__main__":
    get_dataset = JsonToDataset()
    train_set = get_dataset(paths["train_path"])
    test_set = get_dataset(paths["test_path"])

    # max_seq_length = get_dataset.get_max_seq_length()
    batch_size = 4

    train_set = MultimodalDataset(train_set)
    test_set = MultimodalDataset(test_set)

    # print(train_set.__getitem__(1))

    train_loader = get_data_loader(train_set, batch_size)
    test_loader = get_data_loader(test_set, batch_size)

    for batched_inputs, labels in train_loader:
        outputs = model(**batched_inputs)
        print(outputs['logits_per_image'].shape)
    