from transformers import CLIPProcessor, CLIPModel

model_name = "patrickjohncyh/fashion-clip"
clip_processor = CLIPProcessor.from_pretrained(model_name)


def train_step(model, device, batch_data, return_loss):
    _, batch_images, batch_texts, batch_labels = batch_data
    inputs = clip_processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)

    return model(return_loss=return_loss, **inputs)


def get_model(device):
    return CLIPModel.from_pretrained(model_name).to(device)
