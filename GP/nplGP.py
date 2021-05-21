import math
import torch
from matplotlib import pyplot as plt
import torch.distributions.dirichlet as d
import numpy as np



class WeightedGP(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.lengthscale = torch.nn.Parameter(torch.ones(1))
        self.sigma_n = torch.nn.Parameter(torch.ones(1))


    
    def forward(self, x, sq_dists=None, weights=None):
        if sq_dists is None:
            #xfull = x.unsqueeze(1)
            sq_dists = torch.pow(torch.cdist(x, x), 2)
        
        K = torch.exp(-0.5*sq_dists/torch.pow(self.lengthscale, 2))
        
        if weights is not None:
            W = torch.diag(torch.sqrt(weights))
            K = torch.matmul(W, torch.matmul(K, W))
        K = K + torch.pow(self.sigma_n, 2) * torch.eye(K.shape[0])
        return K
    
    def predict(self, xtrain, ytrain, xtest, train_weights=None, KK=None, return_diag=True):
        if train_weights is not None:
            W = torch.diag(torch.sqrt(train_weights))
            ytrain= torch.matmul(W, ytrain)
            
        if KK is None:
            KK = get_kernel(xtrain, xtrain, self.lengthscale)
            
            if train_weights is not None:
                KK = torch.matmul(W, torch.matmul(KK, W))
            KK += torch.pow(self.sigma_n, 2) * torch.eye(KK.shape[0])
            
        Kk = get_kernel(xtrain, xtest, self.lengthscale)
        if train_weights is not None:
            Kk = torch.matmul(W, Kk)

        kK = Kk.transpose(0, 1)
        kk = get_kernel(xtest, xtest, self.lengthscale)
        
        M = torch.inverse(KK)
        #kK = torch.transpose(Kk, 0, 1)
        
        A = torch.matmul(kK, M)
        mean = torch.matmul(A, ytrain)
        cov = torch.matmul(A, Kk)
        if return_diag:
            cov = torch.diag(kk) - torch.diag(cov)
        else:
            cov = kk - cov
            
        return mean.detach().numpy(), cov.detach().numpy() #mean.item(), cov.item()

    def sample(self, xtrain, ytrain, xtest, num_samples, train_weights=None):
        mean, cov = self.predict(xtrain, ytrain, xtest, train_weights=train_weights, return_diag=False)
        
        samp = np.random.multivariate_normal(mean.squeeze(), cov, size=num_samples)
        return samp
        
    
    def plot(self, xtrain, ytrain, xtest, mean, cov):

        var = torch.diag(cov) + mean.squeeze()
        #pdb.set_trace()
        sd = np.sqrt(var.detach().numpy())
        m = mean.squeeze().detach().numpy()
        
        plt.plot(xtest.detach().numpy(), m)
        plt.plot(xtest.detach().numpy(), m+sd, 'k')
        plt.plot(xtest.detach().numpy(), m-sd, 'k')
        plt.plot(xtrain, ytrain, 'x')
        plt.show()
        
def get_kernel(x, y, lengthscale):
    sq_dists = torch.pow(torch.cdist(x, y), 2)
    K = torch.exp(-0.5 * sq_dists / torch.pow(lengthscale, 2))
    return K

def weighted_likelihood(K, y, weights=None, neg=True):
    if weights is not None:
        W = torch.diag(torch.sqrt(weights))
        y = torch.matmul(W, y)
    M = torch.inverse(K) #Y TODO Cholesky
    ll = -0.5*torch.matmul(torch.transpose(y, 0, 1), torch.matmul(M, y)) -0.5*torch.logdet(K) #skipping const term
    if neg:
        ll = -1*ll
    return ll


def train_GP_weighted(xtrain, ytrain, xtest=None, return_mean=True, num_samples=0, num_iters=5000, weights=None, verbose=True):
    '''
    Learns the max likelihood lengthscale and noise term for a squared exponential Gaussian process, given optional weights.
    xtrain: NxD torch tensor of training covariates
    ytrain: Nx1 torch tensor of training targets
    xtest (optional): MxD torch tensor of test covariates
    return_mean: Boolean, if true and xtest is not None, includes mean and standard deviation of predictions at xtest
    num_samples: Integer (>=0), if non-zero and xtest is not None, includes return_samples samples from the posterior
    num_iters: number of iterations
    weights: Size-N torch tensor of weights, corresponding to the implied weight of each observation (default=1)

    returns: Dict with following entries:
    'lengthscale': Inferred lengthscale
    'sigma_n': Inferred noise standard deviation
    'mean': Mean function at xtest (only included if xtest is not None and return_mean=True)
    'std': Standard deviation at xtest (only included if xtest is not None and return_mean=True)
    'samples': num_samples samples from posterior (only included if xtest is not None and num_samples > 0)
    '''

    model = WeightedGP()
    optimizer = torch.optim.Adam(model.parameters())
    for t in range(num_iters):
        K = model(xtrain, weights=weights)
        loss = weighted_likelihood(K, ytrain, weights=weights)
        if t % 1000 == 0:
            if verbose:
                print('iter {}, lengthscale={}, noise={}, loss={}'.format(t, model.lengthscale.item(), 
                                                                          model.sigma_n.item(), loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    res = {'lengthscale': np.abs(model.lengthscale.item()), 'sigma_n': np.abs(model.sigma_n.item())}
    if xtest is not None:
        if return_mean:
            mean, sigmas  = model.predict(xtrain, ytrain, xtest, train_weights=weights)
            sigmas = np.sqrt(sigmas)
            res['mean'] = mean
            res['std'] = sigmas
        if num_samples > 0:
            res['samples'] = model.sample(xtrain, ytrain, xtest, num_samples, train_weights=weights)
    return res
    
    
def train_GP_MLE(xtrain, ytrain, xtest=None, return_mean=True, num_samples=0, num_iters=5000, verbose=True):
    '''
    Learns the max likelihood lengthscale and noise term for a squared exponential Gaussian process, given optional weights.
    xtrain: NxD torch tensor of training covariates
    ytrain: Nx1 torch tensor of training targets
    xtest (optional): MxD torch tensor of test covariates
    return_mean: Boolean, if true and xtest is not None, includes mean and standard deviation of predictions at xtest
    num_samples: Integer (>=0), if non-zero and xtest is not None, includes return_samples samples from the posterior
    num_iters: number of iterations
    weights: Size-N torch tensor of weights, corresponding to the implied weight of each observation (default=1)

    returns: Dict with following entries:
    'lengthscale': Inferred lengthscale
    'sigma_n': Inferred noise standard deviation
    'mean': Mean function at xtest (only included if xtest is not None and return_mean=True)
    'std': Standard deviation at xtest (only included if xtest is not None and return_mean=True)
    'samples': num_samples samples from posterior (only included if xtest is not None and num_samples > 0)
    '''
    res = train_GP_weighted(xtrain, ytrain, xtest=xtest, return_mean=return_mean, num_samples=num_samples, num_iters=num_iters, weights=None, verbose=verbose)
    return res

    
def train_GP_WLB(xtrain, ytrain, xtest=None, return_mean=True, return_samples=False, num_bootstraps = 1, samples_per_bootstrap=10, num_iters=5000, verbose=True):
    '''
    Samples from the WLB (=NPL with alpha=0)
    xtrain: NxD torch tensor of training covariates
    ytrain: Nx1 torch tensor of training targets
    xtest (optional): MxD torch tensor of test covariates
    return_mean: Boolean, if true and xtest is not None, includes mean and standard deviation of predictions at xtest
    samples_per_bootstrap: Number of samples to generate per bootstrap of predictive distribution (if xtest is not None)
    
    return_samples: Boolean, if true and xtest is not None, includes the raw samples in the return object
    num_bootstraps: Integer, number of bootstrap samples
    samples_per_bootstrap: integer, number of predictive samples to generate per bootstrap if xtest is not None
    num_iters: number of iterations
    

    returns: Dict with following entries:
    'lengthscale': Sampled lengthscales
    'sigma_n': Sampled noise standard deviations
    'mean': Mean function at xtest (only included if xtest is not None and return_mean=True)
    'std': Standard deviation at xtest (only included if xtest is not None and return_mean=True)
    'samples': return_samples samples from posterior (only included if xtest is not None and return_samples=True)
    '''
    weight_generator = d.Dirichlet(torch.ones(len(ytrain)))

    lengthscale = []
    sigma_n = []
    if xtest is not None:
        samples = np.zeros((0, len(xtest)))
    for b in range(num_bootstraps):
        weights = weight_generator.sample() * len(ytrain)
        bres = train_GP_weighted(xtrain, ytrain, xtest=xtest, return_mean=False, num_samples=samples_per_bootstrap, num_iters=5000, weights=weights, verbose=verbose)
        if xtest is not None:
            samples = np.vstack((samples, bres['samples']))
        lengthscale.append(bres['lengthscale'])
        sigma_n.append(bres['sigma_n'])
    res = {'lengthscale': lengthscale, 'sigma_n': sigma_n}
    if return_mean:
        res['mean'] = np.mean(samples, axis=0)
        res['std'] = np.std(samples, axis=0)
    if return_samples:
        res['samples'] = samples
        
    return res

def train_GP_NPL(xtrain, ytrain, x_prior_dist, y_prior_dist, xtest=None, return_mean=True, return_samples=False, num_bootstraps = 1, samples_per_bootstrap=10, num_iters=5000, alpha=1., num_pseudo=10, verbose=True):
    '''
    Samples from the NPL
    xtrain: NxD torch tensor of training covariates
    ytrain: Nx1 torch tensor of training targets
    x_prior_dist: torch prior distribution over covariates
    y_prior_dist: torch prior distribution over targets
    xtest (optional): MxD torch tensor of test covariates
    return_mean: Boolean, if true and xtest is not None, includes mean and standard deviation of predictions at xtest
    samples_per_bootstrap: Number of samples to generate per bootstrap of predictive distribution (if xtest is not None)
    
    return_samples: Boolean, if true and xtest is not None, includes the raw samples in the return object
    num_bootstraps: Integer, number of bootstrap samples
    samples_per_bootstrap: integer, number of predictive samples to generate per bootstrap if xtest is not None
    num_iters: number of iterations
    alpha: concentration parameter (>0)
    num_pseudo: number of pseudo-samples
    

    returns: Dict with following entries:
    'lengthscale': Sampled lengthscales
    'sigma_n': Sampled noise standard deviations
    'mean': Mean function at xtest (only included if xtest is not None and return_mean=True)
    'std': Standard deviation at xtest (only included if xtest is not None and return_mean=True)
    'samples': return_samples samples from posterior (only included if xtest is not None and return_samples=True)
    '''
    dirichlet_weight=torch.cat((torch.ones(len(ytrain)), (alpha/num_pseudo)*torch.ones(num_pseudo)), 0)
    weight_generator = d.Dirichlet(dirichlet_weight)

    lengthscale = []
    sigma_n = []
    if xtest is not None:
        samples = np.zeros((0, len(xtest)))
    for b in range(num_bootstraps):
        weights = weight_generator.sample() * (len(ytrain) + alpha)
        pseudo_x = x_prior_dist.sample(sample_shape=torch.Size([num_pseudo]))
        pseudo_y = y_prior_dist.sample(sample_shape=torch.Size([num_pseudo]))
        both_x = torch.cat((xtrain, pseudo_x), 0)
        both_y = torch.cat((ytrain, pseudo_y), 0)
        
        bres = train_GP_weighted(both_x, both_y, xtest=xtest, return_mean=False, num_samples=samples_per_bootstrap, num_iters=5000, weights=weights, verbose=verbose)
        if xtest is not None:
            samples = np.vstack((samples, bres['samples']))
        lengthscale.append(bres['lengthscale'])
        sigma_n.append(bres['sigma_n'])
    res = {'lengthscale': lengthscale, 'sigma_n': sigma_n}
    if return_mean:
        res['mean'] = np.mean(samples, axis=0)
        res['std'] = np.std(samples, axis=0)
    if return_samples:
        res['samples'] = samples
        
    return res
