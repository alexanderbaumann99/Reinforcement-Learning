from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import torch.nn as nn
import enum
import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import seaborn as sns
import itertools
from torch.distributions import MultivariateNormal
from sklearn import datasets
from torch.utils.tensorboard import SummaryWriter

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
        )
    def forward(self, x):
        return self.net(x)


# --- Modules de base

class FlowModule(nn.Module):
    def __init__(self):
        super().__init__()

    def invf(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns f^-1(x) and log |det J_f^-1(x)|"""
        pass

    def f(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns f(x) and log |det J_f(x)|"""
        pass

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


class AffineCoupling(FlowModule):
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

class FlowModules(FlowModule):
    """A container for a succession of flow modules"""
    def __init__(self, flows):
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

    def modulenames(self, backward=False):
        return [f"L{ix} {module.__class__.__name__}" for ix, module in enumerate(reversed(self.flows) if backward else self.flows)]

    def f(self, x):
        zs, logdet = self.apply(self.flows, lambda m, x: m.f(x), x)
        return zs, logdet

    def invf(self, y):
        zs, logdet = self.apply(reversed(self.flows), lambda m, y: m.invf(y), y)
        return zs, logdet


class FlowModel(FlowModules):
    """Flow model = prior + flow modules"""
    def __init__(self, prior, flows):
        super().__init__(flows)
        self.prior = prior

    def learn(self, x):
        zs, logdet = super().f(x)
        logprob = self.prior.log_prob(zs[-1])
        return logprob, zs, logdet

def collect_flow(flowmodel,n=2000):
  
    flows=flowmodel.flows
    n_layers=len(flows)+1
    n_plots=int(math.ceil(math.sqrt(n_layers)))

    z=flowmodel.prior.sample((n,))
    samples=[]
    new_dict={"prior":z}
    samples.append(new_dict)
    for flow_comp in reversed(flows):
        with torch.no_grad():
            z,_=flow_comp.invf(z)
            z_ap=z.numpy()
        if type(flow_comp)==type(Conv1()):
            new_dict={"Conv1":z_ap}
        elif type(flow_comp)==type(AffineCoupling()):
            new_dict={"AffineCouling": z_ap}
        elif type(flow_comp)==type(ActNorm()):
            new_dict={"ActNorm": z_ap} 
        samples.append(new_dict)

    fig,axs=plt.subplots(n_plots,n_plots,figsize=(50,50))
    a=0
    b=0
    for d in samples:
        name=list(d.keys())[0]
        axs[a,b].scatter(d[name][:,0],d[name][:,1],s=0.7) 
        axs[a,b].set_title(name)
        b+=1
        if b>n_plots-1:
            b=0
            a+=1 
    plt.show()
            





if __name__=="__main__":

    depth=5
    hidden=64
    lr=0.0002
    decay=1e-8
    batch_size=128
    num_epoch=30000
    n_samples=20000
    data = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=0.05,random_state=0)[0].astype(np.float32)
    #data = datasets.make_moons(n_samples=n_samples,shuffle=True,noise=0.05,random_state=0)[0].astype(np.float32)
    loader=iter_data(data,batch_size)
    writer=SummaryWriter()

    conv1 = [Conv1() for i in range(depth)]
    norms = [ActNorm() for _ in range(depth)]
    couplings = [AffineCoupling(hidden=hidden) for _ in range(depth)]
    flows = list(itertools.chain(*zip(norms,conv1,couplings))) # append a coupling layer after each 1x1
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2) )
    model=FlowModel(prior,flows)
    optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=decay)
    print("number of params: ", sum(p.numel() for p in model.parameters()))


    for epoch in range(num_epoch):

        batch=next(loader)
        prior_logprob,_, log_det = model.learn(batch)
        logprob = prior_logprob + log_det  
        loss = -torch.mean(logprob) 
        writer.add_scalar("Loss",loss,epoch)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch%100==0:
            print("--EPOCH\t%d--LOSS\t%.3f--" % (epoch,loss.item()))
        
    n_samples=5000
    #torch.save(model.state_dict(),"./model_moons.txt")
    sample_prior=prior.sample((n_samples,))
    gen_x=model.invf(sample_prior)[0]
    x=datasets.make_circles(n_samples=n_samples,factor=0.5,noise=0.05,random_state=0)[0].astype(np.float32)
    #x = datasets.make_moons(n_samples=n_samples,shuffle=True,noise=0.05,random_state=0)[0].astype(np.float32)
    z = gen_x[-1]
    z = z.detach().numpy()

    fig=plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.scatter(x[:,0], x[:,1], c='b', s=5, alpha=0.5)
    ax2.scatter(z[:,0], z[:,1], c='r', s=5, alpha=0.5)
    ax2.set_title('Generated Data')
    ax.set_title('Real Data')
    plt.show()

   
    collect_flow(model,2000)
