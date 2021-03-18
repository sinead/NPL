import torch
import torch.distributions as D
from torch.utils.data import Dataset
import pdb

class GaussianMixDistribution(D.Distribution):
    def __init__(self, pX=None):
        super().__init__()
        
        #self.flip_var_order = flip_var_order
        #if is_test:
        #self.pX = D.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        #else:
        if pX is None:
            mix = D.Categorical(torch.ones(3,))
            comp = D.Independent(D.Normal(
                torch.randn(3,2), 0.3*torch.ones(3,2)), 1)
            self.pX  = D.MixtureSameFamily(mix, comp)
        else:
            self.pX = pX

        
    def rsample(self, sample_shape=torch.Size()):
        X = self.pX.sample(sample_shape)

        return X

    def log_prob(self, X, Y):
        return self.pX.log_prob(X)


class GAUSSIANMIX(Dataset):
    def __init__(self, dataset_size=50000, pX=None):
        self.input_size = 2
        self.label_size = 1
        self.dataset_size = dataset_size
        self.base_dist = GaussianMixDistribution(pX=pX)
        self.X = self.base_dist.rsample(torch.Size([self.dataset_size]))
        self.weights_dist = D.Exponential(torch.tensor([1.0]))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.X[i], torch.zeros(self.label_size), self.weights_dist.sample()




