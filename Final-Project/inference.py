import torch, torchvision
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from preprocess import JsonToDataset, MultimodalDataset, get_data_loader

# paths
paths = {
    ## dataset & image path
    "train_path": "/kovar-vol/kovar/dataset/train.json",
    "test_path": "/kovar-vol/kovar/dataset/test.json",
    "image_path": "/kovar-vol/images",
    }   

# model_checkpoint = "openai/clip-vit-base-patch32"
model_checkpoint = "koclip/koclip-base-pt"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)


resize_and_normalize = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize((224, 224)),      ## refernece: https://github.com/openai/CLIP/issues/248, https://github.com/openai/CLIP/issues/69
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), ## mean & std value are convention calculated from ImageNet.
                                            torchvision.transforms.ToPILImage(),        ## CLIP model got image input in the type of not tensor but PIL.Image
                                            ])


if __name__ == "__main__":
    
    get_dataset = JsonToDataset()
    # train_set = get_dataset(paths["train_path"])
    test_set = get_dataset(paths["test_path"])

    batch_size = 8

    test_set = MultimodalDataset(test_set, transforms=None)
    test_loader = get_data_loader(test_set, batch_size)

    num_correct = 0
    for batched_inputs, labels in test_loader:
        outputs = model(**batched_inputs)
        logits_per_image = outputs['logits_per_image']
        
        ## 1) get proper outputs from 4*12 tensor by proper indexing
        indices = torch.arange(logits_per_image.shape[1]).reshape(batch_size,3)      ## indices: torch.tensor([[0, 1, 2],[3,4,5],[6,7,8],[9,10,11]])
        logits_per_image = torch.gather(input=logits_per_image,dim= 1, index = indices)

        # print(logits_per_image)
        
        ## 2) convert logits_per_image outputs into one-hot like 
        one_hot_outputs = F.one_hot(logits_per_image.argmax(dim=1), num_classes=3).detach().numpy()
        
        ## 3) calculate the right outpus comparing to labels
        num_correct_in_batch = (one_hot_outputs == labels).all(axis=1).sum()
        print(num_correct)
        num_correct += num_correct_in_batch
        print(num_correct_in_batch, num_correct)
        
    print(f'accuracy: {num_correct / len(test_set)}')
        
            