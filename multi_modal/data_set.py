import torch
import open_clip

from transformers import CLIPProcessor
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(interpolation=Image.BICUBIC),
    transforms.PILToTensor(),
])

model_name = "patrickjohncyh/fashion-clip"
clip_processor = CLIPProcessor.from_pretrained(model_name)

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
#
#
# class CustomDatasetWithPreprocessor(Dataset):
#     def __init__(self, item_no_list, image_paths, sentences, labels):
#         self.item_no_list = item_no_list
#         self.image_paths = image_paths
#         self.sentences = tokenizer(sentences)
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         filename = self.image_paths[idx].replace('https://img.29cm.co.kr', '').rpartition('/')[-1]
#         image = Image.open(f'../data/product/images/{filename}').convert("RGB").resize((224, 224))
#
#         image = preprocess(Image.open(image))
#         sentence = self.sentences[idx]
#
#         return self.item_no_list[idx], image, sentence, torch.tensor(self.labels[idx], dtype=torch.long)


class CustomDatasetWithProcessor(Dataset):
    def __init__(self, item_no_list, image_paths, sentences, max_length, labels):
        self.item_no_list = item_no_list
        self.image_paths = image_paths
        self.sentences = sentences
        self.labels = labels
        self.agument = transforms.TrivialAugmentWide(interpolation=Image.BICUBIC)
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        filename = self.image_paths[idx].replace('https://img.29cm.co.kr', '').rpartition('/')[-1]
        image = Image.open(f'../data/product/images/{filename}').convert("RGB").resize((224, 224))

        inputs = clip_processor(
            images=self.agument(image),
            text=self.sentences[idx],
            return_tensors="pt",
            padding=True
        )

        inputs['input_ids'] = inputs['input_ids'][0]  # (S,)
        inputs['attention_mask'] = inputs['attention_mask'][0]  # (S,)
        inputs['pixel_values'] = inputs['pixel_values'][0]  # (C, H, W)

        return inputs


class CustomDatasetWithOutProcessor(Dataset):
    def __init__(self, item_no_list, image_paths, sentences, labels):
        self.item_no_list = item_no_list
        self.image_paths = image_paths
        self.sentences = sentences
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        filename = self.image_paths[idx].replace('https://img.29cm.co.kr', '').rpartition('/')[-1]
        image = Image.open(f'../data/product/images/{filename}').convert("RGB").resize((224, 224))
        image = self.transform(image)

        return self.item_no_list[idx], image, self.sentences[idx], torch.tensor(self.labels[idx], dtype=torch.long)


def print_dataset_names():
    print('CustomDatasetWithProcessor', 'CustomDatasetWithOutProcessor', 'CustomDatasetWithPreprocessor')


def get_dataset(item_no_list, image_paths, sentences, labels, dataset_name=''):
    max_length = max(len(sentence) for sentence in sentences)

    dataset_dict = {
        'CustomDatasetWithProcessor': CustomDatasetWithProcessor(item_no_list, image_paths, sentences, max_length, labels),
        'CustomDatasetWithOutProcessor': CustomDatasetWithOutProcessor(item_no_list, image_paths, sentences, labels),
        # 'CustomDatasetWithPreprocessor': CustomDatasetWithPreprocessor(item_no_list, image_paths, sentences, labels)
    }

    return dataset_dict[dataset_name]
