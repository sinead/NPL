import torch
import torch.distributions as D
from torch.utils.data import Dataset
import pdb

class SynthDistribution(D.Distribution):
    def __init__(self, is_test):
        super().__init__()
        #self.flip_var_order = flip_var_order
        #if is_test:
        #self.pX = D.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        #else:
        mix = D.Categorical(torch.ones(2,))
        comp = D.Uniform(torch.tensor([0.0, 0.35]), torch.tensor([0.45, 1.0]))
        self.pX  = D.MixtureSameFamily(mix, comp)
        self.pY1 = D.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.pY2 = lambda X: D.Normal(torch.sin(10*X), 0.05)
        #self.pY2 = lambda X, Y1: D.Normal(X-Y1, 0.05)


    def rsample(self, sample_shape=torch.Size()):
        X = self.pX.sample(sample_shape)
        X = X.view(X.shape[0], -1)
        Y1 = self.pY1.sample(sample_shape)
        Y2 = self.pY2(X).sample()

        Y = torch.stack((Y1, Y2), dim=-1).squeeze()
        return X, Y

    def log_prob(self, X, Y):
        return self.pY(X).log_prob(Y)


class SYNTH(Dataset):
    def __init__(self, dataset_size=50000, is_test=False):
        self.input_size = 2
        self.label_size = 1
        self.dataset_size = dataset_size
        self.base_dist = SynthDistribution(is_test=is_test)
        self.X, self.Y = self.base_dist.rsample(torch.Size([self.dataset_size]))
        self.X = self.X.view(self.X.shape[0], -1)
        self.weights_dist = D.Exponential(torch.tensor([1.0]))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.Y[i], self.X[i], self.weights_dist.sample()




