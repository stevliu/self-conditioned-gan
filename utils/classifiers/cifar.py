import sys
sys.path.append('utils/classifiers')

from pytorch_playground.cifar.model import cifar10

class Classifier():
    def __init__(self):
        self.classifier = cifar10().cuda()

    def get_predictions(self, x):
        assert(x.size(1) == 3)
        return self.classifier(x).argmax(dim=1)
