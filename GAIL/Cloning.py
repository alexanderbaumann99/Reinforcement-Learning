import argparse
from cmath import exp
from posixpath import expanduser
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
import pickle




class Cloning(object):
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
    
        self.ob_dim=env.observation_space.shape[0] 

        self.actor=nn.Sequential(
            nn.Linear(self.ob_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,self.action_space.n),
            nn.Softmax(dim=-1)
        )
        
        self.actor_old=copy.deepcopy(self.actor)
        self.actor.to(self.device)
        

        self.lr_a=opt.lr_a
        
        self.optimizer_actor  = torch.optim.Adam(self.actor.parameters(),self.lr_a)
    
        self.actor_count=0

        self.loadExp("expert.pkl")

       


    def loadExp(self,file):
        with open(file,"rb") as handle:
            expert_data=torch.FloatTensor(pickle.load(handle)).to(self.device)
            self.exp_states=expert_data[:,:self.ob_dim].contiguous()
            self.exp_actions=expert_data[:,self.ob_dim:].contiguous()

    def toOneHot(self,actions):
        actions=torch.LongTensor(actions.view(-1))
        oneHot=torch.FloatTensor(torch.zeros(actions.size()[0],self.action_space.n,device=self.device))
        oneHot[range(actions.size()[0]),actions]=1
        return oneHot

    def toIndex(self,Hot):
        
        ac=torch.LongTensor(torch.arange(0,self.action_space.n)).view(1,-1)
        ac=ac.expand(Hot.size()[0],-1).contiguous().view(-1)
        actions=ac[Hot.view(-1)>0].view(-1).to(self.device)

        return actions
        

    def act(self, obs):

        prob=self.actor(torch.FloatTensor(obs).to(self.device))
        dist=Categorical(prob)
        
        action=dist.sample()
       
        return action.item()

    def learn(self):

        actions=self.toIndex(self.exp_actions).view(-1,1).to(torch.int64)
        states=self.exp_states
        prob=self.actor(states)
        log_prob= - torch.log(prob)

        log_prob_act= log_prob.gather(1,actions).view(-1).mean(-1)

        self.optimizer_actor.zero_grad()
        log_prob_act.backward()
        self.optimizer_actor.step()
        self.actor_loss=log_prob_act.item()
       
    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_obs, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
          
    

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
    
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('./config_GAIL.yaml', "Cloning")
    
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]   
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = Cloning(env,config)

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
           
            new_obs = agent.featureExtractor.getFeatures(new_obs)
            agent.store(ob, action, new_obs, reward, done,j)
            
            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
                logger.direct_write("actor loss", agent.actor_loss, agent.actor_count)
                
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0

                break

                       
      
    env.close()
