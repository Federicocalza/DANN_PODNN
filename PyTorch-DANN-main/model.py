import torch.nn as nn
import torch.nn.functional as F
import torch
import podnn_torch
from utils import GradReverse


'''
Here are the three modules, the encoder is the feature extractor that is connected to the classifier, which does supervised training on the source domain and the domain classifier that has a PODNN pipeline (the different part of this are explained in the PODNN class file).
'''
class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        self.name= 'encoder'
        conv_dim = 32

        self.conv1 = nn.Conv2d(args.channels, conv_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv4 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(conv_dim * 2, conv_dim * 2, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(conv_dim * 4, conv_dim * 4, kernel_size=3, padding=1)
        self.pool9 = nn.MaxPool2d(2, stride=2)

        self.flat_dim = 4 * 4 * conv_dim * 4
        self.fc1 = nn.Linear(self.flat_dim, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool6(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool9(x)

        x = x.view(-1, self.flat_dim)
        x = F.relu(self.fc1(x))
        return x


class classifier(nn.Module):
    def __init__(self, args):
        super(classifier, self).__init__()
        self.name='classifier'
        self.fc1 = nn.Linear(128, args.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        CheckFiniteLayer(x)
        return x


class discriminator(nn.Module):
    def __init__(self, args):
        super(discriminator, self).__init__()
        self.name='discriminator'
        self.args = args

        self.l1 = nn.Sequential(nn.Linear(128, 500),
                           nn.LeakyReLU(negative_slope = 0.2))
                           
        self.l2 =nn.Sequential(nn.Linear(500, 500),                               
                          nn.LeakyReLU(negative_slope = 0.2))
        self.l3 = nn.Linear(500, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        
        self.multi_branched_discriminator = nn.Sequential(
            podnn_torch.InputLayer(n_models=self.args.num_branches),
            podnn_torch.ParallelLayer(self.l1),
            podnn_torch.OrthogonalLayer1D(),
            podnn_torch.ParallelLayer(self.l2),
            podnn_torch.OrthogonalLayer1D(),
            podnn_torch.AggregationLayer(stride=2,input_dim=500),
            nn.Linear(in_features=podnn_torch.agg_out_dim,out_features=1),
        nn.Sigmoid())

           

    def forward(self, x, alpha=-1):
        if self.args.method.lower() == 'dann':
            x = GradReverse.apply(x, alpha) #This function chenges the sign of the gradient during backprop so that the features become domain invariant
        x = self.multi_branched_discriminator(x)        
        return x
    

class CheckFiniteLayer(nn.Module):
    def __init__(self, layer_name):
        super(CheckFiniteLayer,self).__init__()
        self.layer_name = layer_name
    
    def forward(self,x):
        if (x < 0).any() or (x > 1).any():
            print(f'Errore NaN rilevato nel layer {self.layer_name}')
            print(x)
        return x
    

            
