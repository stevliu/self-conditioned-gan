import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision import datasets
from torch.nn import functional as F
from torchvision import transforms

CLASSIFIER_PATH = 'mnist_model.pt'

class Classifier():
    def __init__(self):
        self.mnist = MNISTClassifier().cuda()

        try:
            self.mnist.load(CLASSIFIER_PATH)
        except Exception as e:
            print(e)
            self.mnist.train()
        

    def get_predictions(self, x):
        assert(x.size(1) == 3)
        result = self.mnist.get_predictions(x[:, 0, :, :])
        for channel_number in range(1, 3):
            result = result + self.mnist.get_predictions(x[:, channel_number, :, :]) * 10**channel_number
        return result

def get_mnist_dataloader(batch_size=100):
    dataset = datasets.MNIST('data/MNIST', train=True, transform=transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, ), (0.5, ))
                                ]))

    return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=12,
                shuffle=True,
                pin_memory=True,
                sampler=None,
                drop_last=True)

class MNISTClassifier(nn.Module):
    def __init__(self, input_dims=1024, n_hiddens=[256, 256], n_class=10):
        super(MNISTClassifier, self).__init__()
        self.input_dims = input_dims
        
        current_dims = input_dims
        layers = OrderedDict()
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

    def get_predictions(self, input):
        logits = self.forward(input)
        return logits.argmax(dim=1)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print('Loaded pretrained MNIST classifier')

    def train(self):
        print('Training MNIST classifier')
        dataloader = get_mnist_dataloader()        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        for epoch in range(10):
            for it, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                logits = self.forward(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                if it % 100 == 0:
                    acc = (self.get_predictions(x) == y).float().mean().item()
                    print(f'[{epoch}, {it}], closs={loss}, acc={acc}')
        

        torch.save(self.state_dict(), CLASSIFIER_PATH)
        

if __name__ == '__main__':
    classifier = Classifier()
    train_loader = get_mnist_dataloader(10)  
    xs, ys = [], []
    for i, (x, y) in enumerate(train_loader):
        if i == 3:
            break  
        xs.append(x.cuda())
        ys.append(y)
    print(ys)
    print(classifier.get_predictions(torch.cat(xs, dim=1)))