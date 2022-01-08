import argparse
import sys
import matplotlib
from numpy import dtype, log
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
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256 , 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.uniform_(m.bias,-0.001,0.001)


class DDPG(object):
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
        
        self.ob_dim=env.observation_space.shape[0]
        self.action_size=env.action_space.shape[0]
        self.action_scale=env.action_space.high[0]
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.actor_local = Actor(self.ob_dim, self.action_size,self.action_scale).to(self.device)
        self.actor_target = copy.deepcopy(self.actor_local)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=opt.lr_a)

        self.critic_local = Critic(self.ob_dim, self.action_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_local)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=opt.lr_c)

        self.noise = Orn_Uhlen(self.action_size, mu=opt.mu, theta=opt.theta, sigma=opt.sigma)

        self.actor_loss=None
        self.critic_loss=None

        self.K=opt.K_epochs
        self.discount=opt.discount
        self.rho=opt.rho 
        self.batch_size=opt.batch_size 

        self.memory=Memory(mem_size=100000)

       
   
    def act(self, obs):
        self.actor_local.eval()
        prob=self.actor_local(torch.FloatTensor(obs).to(self.device).view(1,-1))
        self.actor_local.train()
        noise=self.noise.sample().to(self.device)
        action=torch.clamp(prob+noise, self.action_space.low[0], self.action_space.high[0])  
        action=action.cpu().data.numpy().flatten()    

        return action

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
        
    def learn(self):

        for _ in range(self.K):

            _,_,batch=self.memory.sample(self.batch_size)

            batch=list(zip(*batch))
            ob=torch.FloatTensor(batch[0]).view((self.batch_size,self.ob_dim)).to(self.device)
            act=torch.FloatTensor(batch[1]).view(self.batch_size,self.action_size).to(self.device)
            rew=torch.FloatTensor(batch[2]).view(-1).to(self.device)
            new_ob=torch.FloatTensor(batch[3]).view((self.batch_size,self.ob_dim)).to(self.device)
            d=torch.FloatTensor(batch[4]).view(-1).to(self.device)

            new_act=self.actor_target(new_ob).view(self.batch_size,self.action_size)
            new_Q=self.critic_target(new_ob,new_act).view(-1)
            target=rew+self.discount*(1-d)*new_Q
            target=target.detach()

            Q=self.critic_local(ob,act).view(-1)
            critic_loss=F.mse_loss(Q,target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_loss=critic_loss.item()

            act_pred=self.actor_local(ob).view(self.batch_size,self.action_size)
            actor_loss= -self.critic_local(ob, act_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()   
            self.actor_optimizer.step()
            self.actor_loss=actor_loss.item()

            for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param.data.copy_((1-self.rho)*local_param.data + self.rho*target_param.data)

            for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
                target_param.data.copy_((1-self.rho)*local_param.data + self.rho*target_param.data)
                

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
        return self.nbEvents>=self.batch_size and self.nbEvents%self.opt.freqOptim==0
        #return done


if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_DDPG.yaml', "DDPG")
    
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]   
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = DDPG(env,config)
  
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
        #agent.nbEvents = 0
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
            action= agent.act(ob)
            new_obs, reward, done, _ = env.step(action)
            #reward/=1000

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
                logger.direct_write("actor loss", agent.actor_loss, agent.nbEvents)
                logger.direct_write("critic loss", agent.critic_loss, agent.nbEvents)

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0
                agent.noise.reset()
                break

        if time.time()>=t+1680:
            break
    
     
    env.close()
