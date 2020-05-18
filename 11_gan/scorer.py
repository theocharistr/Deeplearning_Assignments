import numpy as np

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torchvision
from torchvision import transforms

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
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

    @classmethod
    def mnist_pretrained(cls):
        model = cls(input_dims=784, n_hiddens=[256, 256], n_class=10)
        m = torch.load('./mnist-classifier.pth', map_location=lambda storage, loc: storage)
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
        return model
        

class Scorer(nn.Module):
    def __init__(self):
        super(Scorer, self).__init__()
        self.model = MLP.mnist_pretrained()

    def forward(self, images):
        #min_quality = -0.9  # This is the score for Gaussian random noise
        min_quality = np.log(0.1)
        max_quality = -0.2459  # This is the score for MNIST
        
        with torch.no_grad():
            self.model.eval()
            scores = self.model.forward(images)
            p_yx = F.softmax(scores, 1)
            p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
            KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
            quality_and_diversity = KL_d.mean()
            quality = (p_yx * (torch.log(p_yx))).sum(dim=1).mean()
            diversity = (p_y * (torch.log(p_y))).mean()
            #return quality, diversity, torch.exp(quality_and_diversity)
            
            score = (quality - min_quality) / (max_quality - min_quality)
            return score


# This function computes the accuracy on the test dataset
def compute_accuracy(net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def check_accuracy(device):
    # This test shows that the classifier does not need normalization of the inputs
    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor
        #transforms.Normalize((0.5,), (0.5,))
    ])

    data_dir = '../data/mnist'
    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    batch_size = 100
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    model = MLP.mnist_pretrained()
    model.to(device)
    acc = compute_accuracy(model, dataloader, device)
    print('Accuracy:', acc)