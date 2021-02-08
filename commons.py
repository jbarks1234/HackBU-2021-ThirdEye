import io
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models
import torchvision.transforms as transforms


def get_model():
    num_classes = 2
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    model.load_state_dict(torch.load("final_model.pt"))
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# ImageNet classes are often of the form `can_opener` or `Egyptian_cat`
# will use this method to properly format it so that we get
# `Can Opener` or `Egyptian Cat`
def format_class_name(class_name):
    class_name = class_name.replace('pos', 'Hard Hat')
    class_name = class_name.replace('hat', 'No Hard Hat')
    class_name = class_name.title()
    return class_name