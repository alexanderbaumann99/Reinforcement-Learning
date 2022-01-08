import argparse
import sys
import matplotlib
from numpy import dtype, log
from torch import distributions
from torch.nn.modules import loss
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
import gym
import gridworld
import torch
from utils import *
from core import *
from memory import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
import copy
from torch.distributions import Categorical,Normal
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim,action_dim,max_action,min_action,hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.action_scale = (max_action - min_action) / 2.0
        self.action_bias = (max_action + min_action) / 2.0

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def gaussian_sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias
        eval_action=self.action_scale * mean + self.action_bias


        # # Enforcing Action Bound
        log_prob = dist.log_prob(x_t)
        log_prob = torch.sum(log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=1, keepdim=True)

        return action, log_prob,eval_action

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=256):
        super(QNetwork, self).__init__()
        self.l1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2_1 = nn.Linear(hidden_dim, hidden_dim)
        #self.l3_1 = nn.Linear(hidden_dim, hidden_dim)
        self.l4_1 = nn.Linear(hidden_dim, 1)

        self.l1_2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2_2 = nn.Linear(hidden_dim, hidden_dim)
        #self.l3_2 = nn.Linear(hidden_dim, hidden_dim)
        self.l4_2 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x0 = torch.cat((s, a), dim=1)

        x1 = F.relu(self.l1_1(x0))
        x1 = F.relu(self.l2_1(x1))
        #x1 = F.leaky_relu(self.l3_1(x1))
        x1 = self.l4_1(x1)

        x2 = F.relu(self.l1_2(x0))
        x2 = F.relu(self.l2_2(x2))
        #x2 = F.leaky_relu(self.l3_2(x2))
        x2 = self.l4_2(x2)
        return x1, x2


class SAC(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        min_action = env.action_space.low[0]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policynetwork = PolicyNetwork(state_dim,action_dim,max_action,min_action,hidden_dim=opt.hidden_dim).to(self.device)
        self.optimizer_policy = optim.Adam(self.policynetwork.parameters(), lr=opt.lr_p)

        self.qnetwork = QNetwork(state_dim, action_dim,hidden_dim=opt.hidden_dim).to(self.device)
        self.target_qnetwork = QNetwork(state_dim, action_dim,hidden_dim=opt.hidden_dim).to(self.device)
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.optimizer_qnetwork = optim.Adam(self.qnetwork.parameters(), lr=opt.lr_c)

        self.memory=Memory(int(10e5))
        
        self.discount=opt.discount
        self.tau=opt.tau
        self.K_epochs=opt.K_epochs

        self.startsteps=opt.startsteps

        self.adaptive=opt.adaptive
        self.alpha=torch.tensor(opt.alpha)

        if self.adaptive is True:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            #self.target_entropy=opt.target_entropy
            #self.log_alpha = torch.tensor(np.log(opt.alpha), requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=opt.lr_alpha)
            self.alpha=self.log_alpha.exp()


        self.noise=Orn_Uhlen(env.action_space.shape[0],sigma=opt.sigma)

    def act(self, obs):

        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if self.test:
            _, _,action = self.policynetwork.gaussian_sample(state)
        else:
            action,_,_ = self.policynetwork.gaussian_sample(state)
            noise=self.noise.sample().to(self.device)
            action=torch.clamp(action+noise, self.action_space.low[0], self.action_space.high[0])  


        return action.cpu().data.numpy().flatten()

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
        
    def learn(self):

        for _ in range(self.K_epochs):

            _,_,batch=self.memory.sample(self.opt.batch_size)
            state, action, reward, next_state, done = list(zip(*batch))
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.Tensor(done).unsqueeze(1).to(self.device)

            with torch.no_grad():
                next_action, next_log_prob,_ = self.policynetwork.gaussian_sample(next_state)
                target_Q1, target_Q2 = self.target_qnetwork(next_state, next_action)
                target_Q = reward + (1-done) * self.discount * (torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob)

            Q1, Q2 = self.qnetwork(state, action)
            Q1_loss = F.mse_loss(Q1, target_Q)
            Q2_loss = F.mse_loss(Q2, target_Q)
            Q_loss = 1/2*(Q1_loss + Q2_loss)
            self.critic_loss=Q_loss.item()

            self.optimizer_qnetwork.zero_grad()
            Q_loss.backward()
            self.optimizer_qnetwork.step()

            action_new, log_prob,_ = self.policynetwork.gaussian_sample(state)
            Q1_new, Q2_new = self.qnetwork(state, action_new)
            policy_loss = (self.alpha.detach()* log_prob - torch.min(Q1_new, Q2_new)).mean()
            self.policy_loss=policy_loss.item()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

                    
            if self.adaptive:
                with torch.no_grad():
                    _,log_prob,_=self.policynetwork.gaussian_sample(state)
                
                alpha_loss = torch.mean(self.alpha * (-log_prob - self.target_entropy) )
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha=self.log_alpha.exp()
                self.alpha_loss=alpha_loss.item()

            for param, target_param in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
                target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)


                
    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_obs, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
           
            tr=(np.squeeze(ob),action,reward,np.squeeze(new_obs),done)
            
            
            #self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.memory.store(tr)
            
    
    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
    
        self.nbEvents+=1
    
        return self.nbEvents%self.opt.freqOptim==0 and self.nbEvents>=self.startsteps 
       

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_SAC.yaml', "SAC")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]   
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = SAC(env,config)
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    t=time.time()
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            #env.render()
            pass

        new_obs = agent.featureExtractor.getFeatures(ob)
        
        while True:
            if verbose:
              pass
                #env.render()

            ob = new_obs
            if agent.nbEvents>=agent.startsteps:
                action= agent.act(ob)
            else:
                action=env.action_space.sample()
            
            new_obs, reward, done, _ = env.step(action)
            reward=reward

            #new_obs = agent.featureExtractor.getFeatures(new_obs)
            agent.store(ob, action, new_obs, reward, done,j)
            
            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
                logger.direct_write("actor loss", agent.policy_loss, agent.nbEvents)
                logger.direct_write("critic loss", agent.critic_loss, agent.nbEvents)
                if agent.adaptive:
                    logger.direct_write("alpha loss", agent.alpha_loss, agent.nbEvents)
                    logger.direct_write("alpha", agent.alpha, agent.nbEvents)

            
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0
                agent.noise.reset()
                break
        
        if time.time()>=t+600:
            #break
            pass
     
    env.close()
