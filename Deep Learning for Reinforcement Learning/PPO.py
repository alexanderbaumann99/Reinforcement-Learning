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




def init_weights(m):

    if isinstance(m,nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.zeros_(m.bias)



class PPO(object):
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
        self.critic=nn.Sequential(
            nn.Linear(self.ob_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.actor_old=copy.deepcopy(self.actor)
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.lr_a=opt.lr_a
        self.lr_c=opt.lr_c
        self.optimizer_actor  = torch.optim.Adam(self.actor.parameters(),self.lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),self.lr_c)

        self.clip=opt.clip
        self.kl=opt.kl
        self.K=opt.K_epochs
        self.beta=opt.beta
        self.delta=opt.delta
        self.eps_clip=opt.eps_clip
        self.discount=opt.discount
        self.gae_lambda=opt.gae_lambda
        self.entropy=opt.entropy
        self.ent_coef=opt.entropy_coef

        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]

        self.actor_count=0
        self.critic_count=0
       


    def act(self, obs):

        prob=self.actor(torch.FloatTensor(obs).to(self.device))
        dist=Categorical(prob)
        
        action=dist.sample()
       
        
        if not self.test:
            self.log_probs.append(dist.log_prob(action))
            self.actions.append(action.detach())
            self.states.append(torch.FloatTensor(obs).to(self.device))
            self.values.append(self.critic(torch.FloatTensor(obs).to(self.device)).detach())
       

        return action.item()

  

    def learn_kl(self):

        
        last_val=self.critic(torch.FloatTensor(self.new_states[-1]).to(self.device)).item()

        returns = np.zeros_like(self.rewards)
        advantage = np.zeros_like(self.rewards)
        adv=0.
        for t in reversed(range(len(self.rewards))):
            if t==len(self.rewards)-1:
                returns[t]=self.rewards[t]+self.discount*(1-self.dones[t])*last_val
                delta = self.rewards[t]+self.discount*(1-self.dones[t])*last_val - self.values[t].item()
            else:
                returns[t]=self.rewards[t]+self.discount*(1-self.dones[t])*returns[t+1]
                delta=self.rewards[t]+self.discount*(1-self.dones[t])*self.values[t+1].item()-self.values[t].item()

            adv=adv*self.discount*self.gae_lambda*(1-self.dones[t])+delta
            advantage[t]=adv

        
                
        target = torch.FloatTensor(returns).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.log_probs, dim=0)).detach().to(self.device)
        
        pi_old=self.actor(old_states).view((-1,self.action_space.n))

        state_value=self.critic(old_states).view(-1) 
        
        for _ in range(self.K):

            probs = self.actor(old_states)
            dist=Categorical(probs)
            log_probs=dist.log_prob(old_actions)

            ratios=torch.exp(log_probs-old_logprobs.detach())
            
            loss1=torch.mean(ratios*advantage.detach())
            loss2=F.kl_div(input=dist.probs,target=pi_old.detach(),reduction='batchmean')
            

            actor_loss=- (loss1-self.beta*loss2)
            if self.entropy:
                entropy=torch.mean(dist.entropy())
                actor_loss-=self.ent_coef*entropy
            self.actor_count+=1
            self.actor_loss=actor_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        probs = self.actor(old_states)
        dist=Categorical(probs)
        DL=F.kl_div(input=dist.probs.view((-1,self.action_space.n)),target=pi_old.view((-1,self.action_space.n)),reduction='batchmean')
        self.kl_loss=DL

        if DL>=1.5*self.delta:
            self.beta*=2
        if DL<=self.delta/1.5:
            self.beta*=0.5

        
        loss=F.smooth_l1_loss(target,state_value.view(-1))
        self.critic_loss=loss
        self.critic_count+=1
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]
        
        
    def learn_ppo(self):

        last_val=self.critic(torch.FloatTensor(self.new_states[-1]).to(self.device)).item()

        returns = np.zeros_like(self.rewards)
        advantage = np.zeros_like(self.rewards)
        adv=0.
        for t in reversed(range(len(self.rewards))):
            if t==len(self.rewards)-1:
                returns[t]=self.rewards[t]+self.discount*(1-self.dones[t])*last_val
                delta = self.rewards[t]+self.discount*(1-self.dones[t])*last_val - self.values[t].item()
            else:
                returns[t]=self.rewards[t]+self.discount*(1-self.dones[t])*returns[t+1]
                delta=self.rewards[t]+self.discount*(1-self.dones[t])*self.values[t+1].item()-self.values[t].item()

            adv=adv*self.discount*self.gae_lambda*(1-self.dones[t])+delta
            advantage[t]=adv

        
                
        target = torch.FloatTensor(returns).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.log_probs, dim=0)).detach().to(self.device)
        
        pi_old=self.actor(old_states).view((-1,self.action_space.n))

        state_value=self.critic(old_states).view(-1) 
        
        for _ in range(self.K):

            probs = self.actor(old_states)
            dist=Categorical(probs)
            log_probs=dist.log_prob(old_actions)

            ratios=torch.exp(log_probs-old_logprobs.detach())
            
            loss1=torch.mean(ratios*advantage.detach())
            
            actor_loss=-loss1
            if self.entropy:
                entropy=torch.mean(dist.entropy())
                actor_loss-=self.ent_coef*entropy
            self.actor_count+=1
            self.actor_loss=actor_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        loss=F.smooth_l1_loss(target,state_value.view(-1))
        self.critic_loss=loss
        self.critic_count+=1
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]
        

    def learn_clip(self):

        
        last_val=self.critic(torch.FloatTensor(self.new_states[-1]).to(self.device)).item()

        returns = np.zeros_like(self.rewards)
        advantage = np.zeros_like(self.rewards)
        adv=0.
        for t in reversed(range(len(self.rewards))):
            if t==len(self.rewards)-1:
                returns[t]=self.rewards[t]+self.discount*(1-self.dones[t])*last_val
                delta = self.rewards[t]+self.discount*(1-self.dones[t])*last_val - self.values[t].item()
            else:
                returns[t]=self.rewards[t]+self.discount*(1-self.dones[t])*returns[t+1]
                delta=self.rewards[t]+self.discount*(1-self.dones[t])*self.values[t+1].item()-self.values[t].item()

            adv=adv*self.discount*self.gae_lambda*(1-self.dones[t])+delta
            advantage[t]=adv

        
                
        target = torch.FloatTensor(returns).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
       
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.log_probs, dim=0)).detach().to(self.device)

        state_values=self.critic(old_states).view(-1)       
        
        pi_old=self.actor(old_states).view((-1,self.action_space.n))

        
        for _ in range(self.K):

            probs = self.actor(old_states)
            dist=Categorical(probs)
            log_probs=dist.log_prob(old_actions)

            ratios=torch.exp(log_probs-old_logprobs.detach())

            loss1=ratios*advantage.detach()
            loss2=torch.clamp(ratios,min=1-self.eps_clip,max=1+self.eps_clip)*advantage.detach()
            actor_loss= -torch.mean(torch.min(loss1,loss2))
            if self.entropy:
                entropy=torch.mean(dist.entropy())
                actor_loss-=self.ent_coef*entropy
            self.actor_count+=1
            self.actor_loss=actor_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        pi_new = self.actor(old_states).view((-1,self.action_space.n))
        DL=F.kl_div(input=pi_new,target=pi_old.view((-1,self.action_space.n)),reduction='batchmean')
        self.kl_loss=DL

        loss=F.smooth_l1_loss(target,state_values)
        self.critic_loss=loss
        self.critic_count+=1
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]

    def learn(self):

        if self.clip==True:
            self.learn_clip()
        elif self.kl:
            self.learn_kl()
        else: 
            self.learn_ppo()
                     

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_obs, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
           
            self.rewards.append(reward)
            self.dones.append(float(done))
            self.new_states.append(new_obs)
    

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
    
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_PPO.yaml', "PPO")
    
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]   
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = PPO(env,config)

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
                logger.direct_write("critic loss", agent.critic_loss, agent.critic_count)
                logger.direct_write("KL loss", agent.kl_loss, agent.critic_count)

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0

                break

                       
      
    env.close()
