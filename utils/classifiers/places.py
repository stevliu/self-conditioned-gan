import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os


class Classifier():
    def __init__(self):
        # the architecture to use
        arch = 'resnet50'

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file,
                                map_location=lambda storage, loc: storage)
        state_dict = {
            str.replace(k, 'module.', ''): v
            for k, v in checkpoint['state_dict'].items()
        }
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()
        self.model = model
        self.mean = [0.485, 0.456, 0.406]
        self.std =  [0.229, 0.224, 0.225]
        self.trn = trn.Normalize(self.mean, self.std)

        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                class_name = line.strip().split(' ')[0][3:]
                classes.append(''.join(class_name.split('/')))
        self.classes = classes

    def get_name(self, id):
        return self.classes[id]
        
    def transform(self, x):
        x = F.interpolate(x, size=(224, 224)) / 255.
        x = torch.stack([self.trn(xi) for xi in x]).cuda()
        return x

    def get_predictions_and_confidence(self, x):
        x = self.transform(x)
        logit = self.model.forward(x)
        values, ind = logit.max(dim=1)
        return ind, values

    def get_predictions(self, x):
        x = self.transform(x)
        logit = self.model.forward(x)
        return logit.argmax(dim=1)

if __name__ == '__main__':
    x = torch.randn((2,3,128,128))
    c = Classifier()
    x = c.get_predictions(x)
    print(x)
