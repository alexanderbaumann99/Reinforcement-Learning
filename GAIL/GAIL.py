import argparse
import sys
from winreg import REG_QWORD
import matplotlib
from numpy import dtype, log
from sklearn.metrics import jaccard_score
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




class GAIL(object):
  

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
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Linear(64,self.action_space.n),
            nn.Softmax(dim=-1)
        )
        self.critic=nn.Sequential(
            nn.Linear(self.ob_dim,64),
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Linear(64,1)
        )
        self.disc=nn.Sequential(
            nn.Linear(self.ob_dim+self.action_space.n,64),
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.disc.to(self.device)

        self.lr_a=opt.lr_a
        self.lr_c=opt.lr_c
        self.lr_d=opt.lr_d

        self.optimizer_actor  = torch.optim.Adam(self.actor.parameters(),self.lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),self.lr_c)
        self.optimizer_disc = torch.optim.Adam(self.disc.parameters(),self.lr_d)

        self.clip=opt.clip
        self.epoch_ppo=opt.epoch_ppo
        self.epoch_disc=opt.epoch_disc
        self.eps_clip=opt.eps_clip
        self.discount=opt.discount
        self.gae_lambda=opt.gae_lambda
        self.entropy=opt.entropy
        self.ent_coef=opt.entropy_coef
        self.max_grad_norm=10

        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]

        self.opt_count=0
        self.traj=0
       
        self.loadExp("expert.pkl")


    def save(self,file):
            torch.save(self.actor.state_dict(), file + "_actor.txt")
            torch.save(self.critic.state_dict(), file + "_critic.txt")  
            torch.save(self.disc.state_dict(), file + "_disc.txt")  

    def load(self,file):
            self.actor.load_state_dict(torch.load(file +"_actor.txt"))
            self.critic.load_state_dict(torch.load(file + "_critic.txt"))
            self.disc.load_state_dict(torch.load(file + "_disc.txt"))


    def loadExp(self,file):
        with open(file,"rb") as handle:
            expert_data=torch.FloatTensor(pickle.load(handle)).to(self.device)
            self.exp_states=expert_data[:,:self.ob_dim].contiguous()
            self.exp_actions=expert_data[:,self.ob_dim:].contiguous()
            self.exp_pairs=torch.cat((self.exp_states,self.exp_actions),1)

    def toOneHot(self,actions):
        actions=torch.cuda.LongTensor(actions.view(-1))
        oneHot=torch.cuda.FloatTensor(torch.zeros(actions.size()[0],self.action_space.n,device=self.device))
        oneHot[range(actions.size()[0]),actions]=1
        return oneHot

    def toIndex(self,Hot):
        ac=torch.cuda.LongTensor.new_empty(size=(1,self.action_space.n))
        ac=ac.expand(Hot.size()[0],-1).contiguous().view(-1)
        actions=ac[Hot.view(-1)>0].view(-1)

        return actions


    def act(self, obs):

        with torch.no_grad():
            prob=self.actor(torch.FloatTensor(obs).to(self.device))
            dist=Categorical(prob)
            
            action=dist.sample()
       
            if not self.test:

                self.log_probs.append(dist.log_prob(action))
                self.actions.append(action)
                self.states.append(torch.FloatTensor(obs).to(self.device))
                self.values.append(self.critic(torch.FloatTensor(obs).to(self.device)))
       

        return action.item()

  
    def learn(self):

        old_states = torch.squeeze(torch.stack(self.states, dim=0)).to(self.device)
        actions = torch.squeeze(torch.stack(self.actions, dim=0)).to(self.device)
        actions_hot=F.one_hot(actions,num_classes=self.action_space.n)
        state_act_pair=torch.cat((old_states,actions_hot),1)
        old_logprobs = torch.squeeze(torch.stack(self.log_probs, dim=0)).to(self.device)
        new_states = torch.squeeze(torch.stack(self.new_states, dim=0)).to(self.device)
        dones = torch.Tensor(self.dones).to(self.device)
        

        with torch.no_grad():
            
            rewards=torch.log(torch.clamp(self.disc(state_act_pair),0.05,0.95)).view(-1)
            """ target_v=torch.zeros(rewards.shape[0]).to(self.device)
            disc=0
            for t in reversed(range(rewards.shape[0])):
                if dones[t]:
                    disc=0
                disc=rewards[t]+self.discount*disc
                target_v[t]=disc
            target_v = (target_v - target_v.mean()) / (target_v.std() + 1e-10) """

            old_values=self.critic(old_states).view(-1)
            new_values=self.critic(new_states).view(-1)
            delta=rewards+self.discount*new_values*(1-dones)-old_values
            advantage1 = copy.deepcopy(delta)
            advantage=advantage1
            n=1
            for t in reversed(range(advantage.shape[0]-1)):
                #dones ??
                if dones[t]:
                    n=1
                else:
                    n+=1
                advantage1[t]=advantage1[t+1]*self.gae_lambda*self.discount*(1-dones[t])+delta[t]
                advantage[t]=advantage1[t]/n

            target_v=advantage+old_values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        
         
        for _ in range(self.epoch_ppo):
            self.optimizer_actor.zero_grad()
            probs = self.actor(old_states)
            dist=Categorical(probs)
            log_probs=dist.log_prob(actions)
            ratios=torch.exp(log_probs-old_logprobs)

            loss1=ratios*advantage
            loss2=torch.clamp(ratios,min=1-self.eps_clip,max=1+self.eps_clip)*advantage

            actor_loss= -torch.mean(torch.min(loss1,loss2))
            if self.entropy:
                entropy=torch.mean(dist.entropy())
                actor_loss-=self.ent_coef*entropy
            
            actor_loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()
            self.actor_loss=actor_loss

            self.optimizer_critic.zero_grad()
            values=self.critic(old_states).view(-1)
            loss=F.mse_loss(values,target_v)
            loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()
            self.critic_loss=loss
        
        for _ in range(self.epoch_disc):
            self.optimizer_disc.zero_grad()
            exp_pred=self.disc(self.exp_pairs)
            act_pred=self.disc(state_act_pair)
            loss_exp=torch.log(1-torch.clamp(exp_pred,0.05,0.95)).view(-1).mean(-1)
            loss_agent=torch.log(torch.clamp(act_pred,0.05,0.95)).view(-1).mean(-1)
            loss_disc=loss_exp+loss_agent
            loss_disc.backward()
            self.optimizer_disc.step()
            self.exp_pred=exp_pred.mean()
            self.act_pred=act_pred.mean()

        self.opt_count+=1
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]

    
                     

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_obs, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
           
            self.rewards.append(reward)
            self.dones.append(done)
            self.new_states.append(torch.FloatTensor(new_obs).to(self.device))
            

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False

        if done:
            self.traj+=1
        self.nbEvents+=1
        """ if self.traj>=5:
            self.traj=0
            return True
        else: 
            return False """
        
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('./config_GAIL.yaml', "GAIL")
    
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]   
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = GAIL(env,config)
    agent.load("./XP/LunarLander-v2/GAIL/save_9000")
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

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("reward/Test", mean / nbTest, itest)
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
                logger.direct_write("loss/actor", agent.actor_loss, agent.opt_count)
                logger.direct_write("loss/critic", agent.critic_loss, agent.opt_count)
                logger.direct_write("disc/expert", agent.exp_pred, agent.opt_count)
                logger.direct_write("disc/actor", agent.act_pred, agent.opt_count)


            if done:
                print("EPISODE %d\t reward=%.3f\t %d actions" %(i,rsum,j))
                logger.direct_write("reward/Train", rsum, i)
                mean += rsum
                rsum = 0

                break

                       
      
    env.close()
