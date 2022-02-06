import itertools
from tkinter.messagebox import NO
from unicodedata import name
from torch.utils.data import DataLoader
import torch.nn as nn
import enum
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import seaborn as sns
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
from sklearn import datasets

sns.set_theme()


def toDataFrame(t: torch.Tensor, origin: str):
    t = t.cpu().detach().numpy()
    df = pd.DataFrame(data=t, columns=(f"x{ix}" for ix in range(t.shape[1])))
    df['ix'] = df.index * 1.
    df["origin"] = origin
    return df



def scatterplots(samples: List[Tuple[str, torch.Tensor]], col_wrap=4):
    """Draw the 

    Args:
        samples (List[Tuple[str, torch.Tensor]]): The list of samples with their types
        col_wrap (int, optional): Number of columns in the graph. Defaults to 4.

    Raises:
        NotImplementedError: If the dimension of the data is not supported
    """
    # Convert data into pandas dataframes
    _, dim = samples[0][1].shape
    samples = [toDataFrame(sample, name) for name, sample in samples]
    data = pd.concat(samples, ignore_index=True)

    g = sns.FacetGrid(data, height=2, col_wrap=col_wrap, col="origin", sharex=False, sharey=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    if dim == 1:
        g.map(sns.kdeplot, "distribution")
        plt.show()
    elif dim == 2:
        g.map(sns.scatterplot, "x0", "x1", alpha=0.6)
        plt.show()
    else:
        raise NotImplementedError()


def iter_data(dataset, bs):
    """Infinite iterator"""
    while True:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        yield from iter(loader)


class MLP(nn.Module):
    """Réseau simple 4 couches"""
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
            nn.Sigmoid()
        )
    def forward(self, x):
        return 10*self.net(x)-5


# --- Modules de base

""" class FlowModule(nn.Module):
    def __init__(self,dim=2):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1,dim,requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1,dim,requires_grad=True))

    def invf(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        #Returns f^-1(x) and log |det J_f^-1(x)|
        x=(y-self.bias)*torch.exp(-self.scale)
        log_det=-self.scale.sum(-1)

        return x,log_det

    def f(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        #Returns f(x) and log |det J_f(x)|
        y=x*torch.exp(self.scale)+self.bias
        log_det=self.scale.sum(-1)
        
        return y,log_det

    def check(self,x,y):
        test_1=     x==self.invf(self.f(x)[0])[0]
        test_2=     y==self.f(self.invf(y)[0])[0]

        if test_1 and test_2:
            return True
        else:
            return False
 """


class DatasetMoons:
    """ two half-moons """
    def sample(self, n):
        moons = datasets.make_circles(n_samples=n,factor=0.5,noise=0.05,random_state=0)[0].astype(np.float32)
        return torch.from_numpy(moons)
   
    


class ActNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, 2, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, 2, requires_grad=True)) 
        self.init_done=False
        
    def f(self,x):
        if not self.init_done:
            self.scale.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.bias.data = (-(x * torch.exp(self.scale)).mean(dim=0, keepdim=True)).detach()
            self.init_done=True

        z = x * torch.exp(self.scale) + self.bias
        log_det = torch.sum(self.scale, dim=1)
        return z, log_det


    def invf (self, z):
        
        x = (z - self.bias) * torch.exp(-self.scale)
        log_det = torch.sum(-self.scale, dim=1)
        return x, log_det


class AffineCoupling(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        self.dim=2
        self.MLP_s=MLP(self.dim//2 ,self.dim-self.dim//2,hidden)
        self.MLP_t=MLP(self.dim//2 ,self.dim-self.dim//2,hidden)
    
    def f(self,x):
        x1=x[:,:self.dim//2]
        x2=x[:,self.dim//2:]
        s=self.MLP_s(x1)
        t=self.MLP_t(x1)
        y1=x1
        y2=x2*torch.exp(s)+t
        y=torch.cat((y1,y2),dim=-1)

        logdet=s.sum(1)

        return y,logdet

    def invf(self,y):

        y1=y[:,:self.dim//2]
        y2=y[:,self.dim//2:]
        s=self.MLP_s(y1)
        t=self.MLP_t(y1)
        x1=y1
        x2=(y2-t)*torch.exp(-s)

        x=torch.cat((x1,x2),dim=-1)

        logdet=(-s).sum(1)
        return x,logdet

class Conv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim=2
        Q = torch.nn.init.orthogonal_(torch.randn(self.dim,self.dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S
        
    def f(self, x):
        W = self._assemble_W()
        y = x @ W
        logdet = torch.sum(torch.log(torch.abs(self.S)))

        return y,logdet 
    
    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def invf(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)+1e-6))
        return x, log_det
""" 
class FlowModules(FlowModule):
    A container for a succession of flow modules
    def __init__(self, flows: FlowModule):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def apply(self, modules_iter, caller, x):
        m, _ = x.shape
        logdet = torch.zeros(m, device=x.device)
        zs = [x]
        for module in modules_iter:
            x, _logdet = caller(module, x)
            zs.append(x)
            logdet += _logdet
        return zs, logdet            

    def modulenames(self, backward=True):
        return [f"L{ix} {module.__class__.__name__}" for ix, module in enumerate(reversed(self.flows) if backward else self.flows)]

    def f(self, x):
        zs, logdet = self.apply(self.flows, lambda m, x: m.f(x), x)
        return zs, logdet

    def invf(self, y):
        zs, logdet = self.apply(reversed(self.flows), lambda m, y: m.invf(y), y)
        return zs, logdet


class FlowModel(FlowModules):
    Flow model = prior + flow modules
    def __init__(self, prior, flows: FlowModule):
        super().__init__(flows)
        self.prior = prior

    def f(self, x):
        # Just computes the prior
        zs, logdet = super().f(x)
        logprob = self.prior.log_prob(zs[-1])
        return logprob, zs, logdet

    def invf(self, x):
        # Just computes the prior
        zs, logdet = super().invf(x)
        logprob = self.prior.log_prob(zs[-1])
        return logprob, zs, logdet

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        _,xs,_ = self.invf(z)
        return xs
 """
class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def f(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.f(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def invf(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.invf(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def f(self, x):
        zs, log_det = self.flow.f(x)
        prior_logprob = self.prior.log_prob(zs[-1])
        return prior_logprob,zs,log_det

    def invf(self, z):
        xs, log_det = self.flow.invf(z)
        return xs, log_det
    
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.invf(z)
        return xs


def train(depth,batch_size,num_epoch,hidden,lr):

    conv1 = [Conv1() for i in range(depth)]
    norms = [ActNorm() for _ in range(depth)]
    couplings = [AffineCoupling(hidden=hidden) for _ in range(depth)]
    flows = list(itertools.chain(*zip(norms, conv1, couplings))) # append a coupling layer after each 1x1

    prior = MultivariateNormal(torch.zeros(2), torch.eye(2) )
    model=NormalizingFlowModel(prior,flows)
    optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-6)
    print("number of params: ", sum(p.numel() for p in model.parameters()))
    
 

    data=DatasetMoons()
    sample=data.sample(2000)
    Loader=DataLoader(sample,batch_size=batch_size,shuffle=True)

    iter=0
    for epoch in range(num_epoch):
            epoch_loss=0
            epoch_iter=0
            for batch in Loader:
                iter+=1
                x=batch
                prior_logprob, zs, log_det = model.f(x)
                logprob = prior_logprob + log_det
                loss = -torch.mean(logprob) 

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss+=loss
                epoch_iter+=1
                """ if iter%100==0:
                    print("--ITER\t%d--LOSS\t%.3f--" % (iter,loss.item())) """

            epoch_loss=epoch_loss/epoch_iter
            print("--EPOCH\t%d--LOSS\t%.3f--" % (epoch,epoch_loss.item()))
            if epoch_loss.item()<0.5:
                break


    zs=model.sample(256)
    x=data.sample(256)
    z = zs[-1]
    z = z.detach().numpy()

    fig=plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.scatter(x[:,0], x[:,1], c='b', s=5, alpha=0.5)
    ax2.scatter(z[:,0], z[:,1], c='r', s=5, alpha=0.5)
    ax2.set_title('z -> x')
    ax.set_title('REAL DATA')
    plt.show()


if __name__=="__main__":

    depth,batch_size,num_epoch,hidden,lr=(5,128,200000,128,1e-4)
    train(depth,batch_size,num_epoch,hidden,lr)
    
