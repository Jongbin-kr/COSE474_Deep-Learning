from transformers import AutoProcessor, AutoModel

from preprocess import JsonToDataset, MultimodalDataset, get_data_loader

# paths
paths = {
    ## dataset & image path
    "train_path": "/kovar-vol/kovar/dataset/train.json",
    "test_path": "/kovar-vol/kovar/dataset/test.json",
    "image_path": "/kovar-vol/images",
    }   

model_checkpoint = "koclip/koclip-base-pt"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)


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