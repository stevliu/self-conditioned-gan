import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os


class Classifier():
    def __init__(self):
        self.model = models.resnet50(pretrained=True).cuda()
        self.model.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.trn = trn.Normalize(self.mean, self.std)

        import json
        with open("utils/classifiers/imagenet_class_index.json") as f:
            self.class_idx = json.load(f)

    def transform(self, x):
        x = F.interpolate(x, size=(224, 224)) / 255.
        x = torch.stack([self.trn(xi) for xi in x]).cuda()
        return x

    def get_name(self, class_id):
        return self.class_idx[str(class_id)][1]

    def get_predictions_and_confidence(self, x):
        x = self.transform(x)
        logit = self.model.forward(x)
        values, ind = logit.max(dim=1)
        return ind, values

    def get_predictions(self, x):
        x = self.transform(x)
        logit = self.model.forward(x)
        return logit.argmax(dim=1)
