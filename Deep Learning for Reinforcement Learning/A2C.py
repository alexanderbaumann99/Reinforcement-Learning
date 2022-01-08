import argparse
import sys
import matplotlib
from numpy import dtype, int64, log
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
from torch.distributions import Categorical
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class PolicyNetwork(nn.Module):
    def __init__(self,ob_dim,n_action):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(ob_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_action)
        self.softmax=nn.Softmax(-1)
        self.act=nn.Tanh() 

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return Categorical(x)


class ValueNetwork(nn.Module):
    def __init__(self,ob_dim):
        super(ValueNetwork,self).__init__()
        self.fc1 = nn.Linear(ob_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x



class A2C(object):
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
      
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        self.actor=PolicyNetwork(env.observation_space.shape[0],env.action_space.n)
        self.critic=ValueNetwork(env.observation_space.shape[0])
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        with torch.no_grad():
            self.critic_target=copy.deepcopy(self.critic) 
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = opt.lr_a)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = opt.lr_c)
                            
        self.memory = deque(maxlen=1000)
    
        self.critic_loss=None
        self.actor_loss=None
        self.counter=0

        self.gae=opt.gae
        self.roll_out=opt.roll_out
        self.discount=opt.discount
        self.gae_lambda=opt.gae_lambda
        self.rho=opt.rho   
        self.ent_coef=opt.entropy_coef
        self.entropy=opt.entropy

        self.entropy_loss=0
        

    def act(self, obs):

        dist=self.actor(torch.FloatTensor(obs).to(self.device))
        action=dist.sample()
        self.logprob=dist.log_prob(action).unsqueeze(0)
        self.entropy_loss+=dist.entropy().mean()
        self.action=action
        self.state=torch.FloatTensor(obs).to(self.device)
                         
        return action.item()

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire

    def learn(self):
        self.learn_1()


    def learn_1(self):
        
        mini_batch = self.memory
        
        obs=list(zip(*mini_batch))[0]
        acts=list(zip(*mini_batch))[1]
        new_obs=list(zip(*mini_batch))[2]
        rewards=list(zip(*mini_batch))[3]
        dones=list(zip(*mini_batch))[4]
        log_prob=list(zip(*mini_batch))[5]


        old_states = torch.squeeze(torch.stack(obs, dim=0)).detach().to(self.device)
        new_states = torch.squeeze(torch.stack(new_obs, dim=0)).detach().to(self.device)
        log_prob = torch.squeeze(torch.stack(log_prob, dim=0)).to(self.device)
       
        if self.roll_out:
           
            returns=self.critic_target(new_obs[-1]).view(1).item()
            Returns = np.zeros(len(rewards))

            for t in reversed(range(len(rewards))):
                returns = rewards[t] + returns * self.discount*(1-dones[t])
                Returns[t]=returns
            
            Returns = torch.FloatTensor(Returns).to(self.device)
        
            values_target=self.critic_target(old_states).view(-1)
            advantage=Returns-values_target 
                  
        elif self.gae:

            Returns = np.zeros_like(rewards)
            returns=self.critic_target(new_obs[-1]).view(1).item()
            values_target=self.critic_target(old_states).view(-1).cpu().detach().numpy()
            advantages = np.zeros_like(rewards)
            advantage=0.
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    returns = rewards[t] + self.discount * (1-dones[t]) *returns
                    td_error = returns - values_target[t]
                else:
                    returns = rewards[t] + self.discount * (1-dones[t]) * returns
                    td_error = rewards[t] + self.discount * (1-dones[t]) * values_target[t+1] - values_target[t]

                Returns[t]=returns
                advantage = advantage * self.gae_lambda * self.discount * (1-dones[t]) + td_error
                advantages[t]=advantage

            advantage = torch.FloatTensor(advantages).to(self.device)
            Returns = torch.FloatTensor(Returns).to(self.device)
            advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        else:
            rewards=torch.Tensor(rewards).to(self.device)
            dones=torch.Tensor(dones).to(self.device)
            values_old=self.critic_target(old_states).view(-1)
            values_new=self.critic_target(new_states).view(-1)
            Returns=rewards+self.discount*values_new*(1-dones)
            advantage=Returns-values_old 
            

        values=self.critic(old_states).view(-1)
        pi_old=self.actor(old_states).probs
        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = F.smooth_l1_loss(values,Returns)
        if self.entropy:
            actor_loss-=self.ent_coef*self.entropy_loss

        self.actor_loss=actor_loss
        self.critic_loss=critic_loss

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        pi_new=self.actor(old_states).probs
        self.kl_loss=F.kl_div(pi_new,pi_old)

        self.counter+=1
        self.memory.clear()
        self.entropy_loss=0
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1-self.rho)*local_param.data + self.rho*target_param.data)



    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_obs, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
           
                   
            new_obs=torch.FloatTensor(new_obs).to(self.device)
            self.memory.append((self.state, self.action,new_obs,reward, float(done),self.logprob))
                
                
    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
      
        return done 

        
if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_A2C.yaml', "RandomAgent")
    
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]   
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = A2C(env,config)

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
        verbose=False

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
            env.render()

        new_obs = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_obs
            
            action= agent.act(ob)
            new_obs, reward, done, _ = env.step(action)

            reward = reward 
            new_obs = agent.featureExtractor.getFeatures(new_obs)

            j+=1
            agent.store(ob, action, new_obs, reward, done,j)

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
                logger.direct_write("Actor Loss", agent.actor_loss, agent.counter)
                logger.direct_write("Critic Loss", agent.critic_loss, agent.counter)
                logger.direct_write("KL Loss", agent.kl_loss, agent.counter)
                

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0

                break
        
        if time.time()>=t+600:
            break
                
      
    env.close()
