import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')


def train_step(model, device, batch_data):
    _, batch_images, batch_texts, batch_labels = batch_data

    batch_images = batch_images.to(device)
    batch_texts = batch_texts.to(device)

    return model(batch_images, batch_texts)


def get_model(device):
    return model.to(device)
