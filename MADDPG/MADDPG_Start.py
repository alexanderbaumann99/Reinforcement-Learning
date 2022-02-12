import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from collections import namedtuple,defaultdict
from utils import *
from core import *
from datetime import datetime
import random
from collections import deque


class Agent(object):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)

    # agent_type: 0 = random, 1 = DDPG, 2 = MADDPG
    def __init__(self,env,opt,id,action_size,obs_shapes,agent_type = 2, nb_subPolicies = 1, optimizer = torch.optim.Adam,lr = 0.1, lrq = 0.1):


        self.id=id
        self.opt=opt


        self.agent_type = agent_type
        self.obs_shapes = obs_shapes
        self.env = env

        print(id,nb_subPolicies)

        self.action_size=action_size
        self.nb_subPolicies=nb_subPolicies
        self.policies = []
        self.targetPolicies = []
        self.q = None
        self.qtarget = None
        self.actor_loss=None
        self.Q_loss=None


        if agent_type==0: #random
            self.nb_subPolicies = 0
        elif agent_type==2:
            self.q=nn.Sequential(
                nn.Linear(sum(obs_shapes)+self.action_size*self.env.n,128),
                nn.LeakyReLU(),
                nn.Linear(128,1)
            )
            self.qtarget=copy.deepcopy(self.q)
            for _ in range(self.nb_subPolicies):
                actor=nn.Sequential(
                    nn.Linear(obs_shapes[id],128),
                    nn.LeakyReLU(),
                    nn.Linear(128,self.action_size),
                    nn.Tanh()
                )
                self.policies.append(actor)
                self.targetPolicies.append(copy.deepcopy(actor))

            self.polyakP = self.opt.polyakP
            self.polyakQ = self.opt.polyakQ
        elif agent_type==1:
            self.q=nn.Sequential(
                nn.Linear(obs_shapes[id]+self.action_size,128),
                nn.LeakyReLU(),
                nn.Linear(128,1)
            )
            self.qtarget=copy.deepcopy(self.q)
            for _ in range(self.nb_subPolicies):
                actor=nn.Sequential(
                    nn.Linear(obs_shapes[id],128),
                    nn.LeakyReLU(),
                    nn.Linear(128,self.action_size),
                    nn.Tanh()
                )
                self.policies.append(actor)
                self.targetPolicies.append(copy.deepcopy(actor))

            self.polyakP = self.opt.polyakP
            self.polyakQ = self.opt.polyakQ
        self.currentPolicy=0

        self.events = [deque(maxlen=self.opt.capacity) for _ in range(nb_subPolicies)]
        self.batchsize = self.opt.batchsize

        if self.opt.fromFile is not None:
            self.load(self.opt.fromFile)


        # Creation optimiseurs
        if agent_type > 0:  # not random
            wdq = self.opt.wdq   # weight decay for q
            wdp = self.opt.wdp  # weight decay for pi
            self.qtarget.load_state_dict(self.q.state_dict())
            for i in range(nb_subPolicies):
                self.targetPolicies[i].load_state_dict(self.policies[i].state_dict())

            self.optimizerP = [optimizer([{"params": self.policies[i].parameters()}], weight_decay=wdp, lr=lr) for i in
                                   range(nb_subPolicies)]
            self.optimizerQ = optimizer([{"params": self.q.parameters()}], weight_decay=wdq, lr=lrq)


    def act(self,obs):

        if self.agent_type == 0:
            a = self.floatTensor.new((np.random.rand(self.action_size) - 0.5) * 2).view(-1)
            return a
        return self.policies[self.currentPolicy](obs)

    def getTargetAct(self,obs):
        if self.agent_type == 0:
            a=self.floatTensor.new((np.random.rand(obs.shape[0],self.action_size) - 0.5) * 2).view(-1,self.action_size)
            return a
        i = np.random.randint(0, self.nb_subPolicies, 1)[0]
        return self.targetPolicies[i](obs)

    def addEvent(self,event):
        if self.agent_type == 0:
            return
        i=self.currentPolicy
        self.events[i].append(event)

    def selectPolicy(self):
        if self.agent_type==0:
            return 0
        i = np.random.randint(0, self.nb_subPolicies, 1)[0]
        self.currentPolicy = i

    def eval(self):
        if self.q is not None:
            for p in self.policies:
                p.eval()
            self.q.eval()
    def train(self):
        if self.q is not None:
            for p in self.policies:
                p.train()
            self.q.train()

    def soft_update(self):
        if self.q is not None:

            for i in range(len(self.policies)):
                for target, src in zip(self.targetPolicies[i].parameters(), self.policies[i].parameters()):
                    target.data.copy_(target.data * self.polyakP + src.data * (1-self.polyakP))

            for target, src in zip(self.qtarget.parameters(), self.q.parameters()):
                target.data.copy_(target.data * self.polyakQ + src.data * (1 - self.polyakQ))

    def setcuda(self,device):
        Agent.floatTensor = torch.cuda.FloatTensor(1, device=device)
        Agent.longTensor = torch.cuda.LongTensor(1, device=device)
        if self.q is not None:
            for p in self.policies:
                p=p.to(device)
            for p in self.targetPolicies:
                p=p.to(device)
            self.q=self.q.to(device)
            self.qtarget=self.qtarget.to(device)

    def save(self,file):

        if self.q is not None:
            for x in range(self.nb_subPolicies):
                torch.save(self.policies[x].state_dict(), file + "_policy_" + str(self.id) + "_" + str(x) + ".txt")
            torch.save(self.q.state_dict(), file + "_value_"+str(self.id)+".txt")

    def load(self,file):
        if self.q is not None:
            for x in range(self.nb_subPolicies):
                self.policies[x].load_state_dict(torch.load(file +"_policy_"+str(self.id)+"_"+str(x)+".txt"))
                self.q.load_state_dict(torch.load(file + "_value_"+str(self.id)+".txt"))


class MADDPG(object):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)
    verbose = 0


    def __init__(self,env,opt,action_size,obs_shapes,noise,noiseTest):
        super(MADDPG, self).__init__()
        self.action_size = action_size
        self.env=env
        self.opt=opt
        self.gamma=self.opt.gamma

        agent_types = self.opt.agent_types
        nb_subPolicies =  self.opt.nb_subPolicies
        lr = self.opt.lr
        lrq = self.opt.lrq
        device = self.opt.device

        nbSteps = self.opt.nbSteps
        freqOptim = self.opt.freqOptim
        optimizer=torch.optim.Adam
        seed=self.opt.seed

        self.freqOptim=freqOptim
        self.nbSteps=nbSteps
        self.noise=noise
        self.noiseTest = noiseTest
        self.startEvents=self.opt.startEvents
        self.test=False

        self.nbEvts=0
        self.nbEvents=0

        self.polyakP=self.opt.polyakP
        self.polyakQ = self.opt.polyakQ
        self.nbOptim=0

        self.nbRuns=0

        self.agents=[]
        for i in range(env.n):
            a=Agent(env,opt,i,action_size,obs_shapes,agent_types[i],nb_subPolicies[i],optimizer,lr=lr[i],lrq=lrq[i])
            self.agents.append(a)

        self.nb=0

        self.sumDiff=0
        prs("lr",lr)

        self.current=[]
        self.batchsize=self.opt.batchsize

        if self.opt.fromFile is not None:
            self.load(self.opt.fromFile)

        if device>=0:
            cudnn.benchmark = True
            torch.cuda.device(device)
            self.device=torch.device('cuda')
            torch.cuda.manual_seed(seed)
            MADDPG.floatTensor = torch.cuda.FloatTensor(1, device=device)
            MADDPG.longTensor = torch.cuda.LongTensor(1, device=device)
            for i in range(env.n):
                self.agents[i].setcuda(device)


    def save(self,file):
        for agent in self.agents:
            agent.save(file)

    def load(self,file):
        for agent in self.agents:
            agent.load(file)


    def store(self,ob,action,new_ob,rewards,done,it):
        d=done[0]
        if it == self.opt.maxLengthTrain:
            print("undone")
            d = False

        for a in self.agents:
            #print(("ob", ob, "a", action, "r", rewards, "new_ob", new_ob, d))
            a.addEvent((ob, action, rewards, new_ob, d))
        self.nbEvts += 1


    def act(self, obs):

        for a in self.agents:
            a.eval()

        if self.nbEvts>self.startEvents:
            with torch.no_grad():

                actions = torch.cat([self.agents[i].act(self.floatTensor.new(obs[i])).view(-1) for i in range(self.env.n)],dim=-1).cpu()
        
                if (not self.test) or self.opt.sigma_noiseTest>0:
                    noise=self.noise
                    if self.test:
                        noise=self.noiseTest
                    e=torch.cat([x.sample() for x in noise],dim=-1)
                    actions=actions+e.view(-1)
                actions=actions.numpy()

        else:
            actions =np.concatenate([x.sample().numpy() for x in self.noise],axis=-1)

        return actions.reshape((self.env.n,-1))


    def endsEpisode(self):
        for i in range(self.env.n):
            self.noise[i].reset()
            self.noiseTest[i].reset()
            self.agents[i].selectPolicy()

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents += 1
        #print(self.nbEvents,self.opt.freqOptim,self.opt.startEvents)
        if self.nbEvents % self.opt.freqOptim == 0 and  self.nbEvents > self.opt.startEvents:
            print("Time to Learn! ")
            return True
        return False

    def learn(self):

        self.nbOptim+=1
        for a in self.agents:
            a.train()
        sl=torch.nn.MSELoss() #SmoothL1Loss()

        for j in range(self.nbSteps):
            for x in range(self.env.n):
                agent=self.agents[x]
                if agent.agent_type==0:
                    continue
                elif len(list(agent.events[agent.currentPolicy]))>=self.batchsize and agent.agent_type==2:
                    batch = random.sample(list(agent.events[agent.currentPolicy]), self.batchsize)
                    batch = list(zip(*batch))

                    state=torch.cuda.FloatTensor(np.array(batch[0]))
                    state_all=state.reshape(self.batchsize,sum(agent.obs_shapes))
                    action=torch.cuda.FloatTensor(np.array(batch[1]))
                    action_all=action.reshape(self.batchsize,self.env.n*self.action_size)
                    rewards=torch.cuda.FloatTensor(np.array(batch[2]))[:,agent.id]
                    new_state=torch.cuda.FloatTensor(np.array(batch[3]))
                    new_state_all=new_state.reshape(self.batchsize,sum(agent.obs_shapes))
                    done=torch.cuda.FloatTensor(np.array(batch[4]))

                    with torch.no_grad():
                        new_actions=torch.zeros((self.batchsize,self.env.n,self.action_size),device=self.device)
                        for i,k_agent in enumerate(self.agents):
                            state_agent_k=new_state[:,k_agent.id,:].view(self.batchsize,-1)
                            new_actions[:,i,:]=k_agent.getTargetAct(state_agent_k)
                        new_actions_all=new_actions.reshape((self.batchsize,self.env.n*self.action_size))
                        target_Q=agent.qtarget(torch.cat((new_state_all,new_actions_all),-1)).view(-1)
                        target=rewards+self.gamma*(1-done)*target_Q
                    
                    agent.optimizerQ.zero_grad()
                    Q_values=agent.q(torch.cat((state_all,action_all),-1)).view(-1)
                    Q_loss=sl(Q_values,target).mean()
                    Q_loss.backward()
                    agent.optimizerQ.step()
                    agent.Q_loss=Q_loss

                    agent.optimizerP[agent.currentPolicy].zero_grad()
                    action_id=agent.policies[agent.currentPolicy](state[:,agent.id,:])
                    action[:,agent.id,:]=action_id
                    action=action.reshape(self.batchsize,self.env.n*self.action_size)
                    actor_loss=agent.q(torch.cat((state_all,action),-1))
                    actor_loss= -(actor_loss).mean(0)
                    actor_loss.backward()
                    agent.optimizerP[agent.currentPolicy].step()
                    agent.actor_loss=actor_loss

                elif len(list(agent.events[agent.currentPolicy]))>=self.batchsize and agent.agent_type==1:
                    
                    batch = random.sample(list(agent.events[agent.currentPolicy]), self.batchsize)
                    batch = list(zip(*batch))

                    state=torch.cuda.FloatTensor(np.array(batch[0]))[:,agent.id,:]
                    action=torch.cuda.FloatTensor(np.array(batch[1]))[:,agent.id,:]
                    rewards=torch.cuda.FloatTensor(np.array(batch[2]))[:,agent.id]
                    new_state=torch.cuda.FloatTensor(np.array(batch[3]))[:,agent.id,:]
                    done=torch.cuda.FloatTensor(np.array(batch[4]))

                    with torch.no_grad():
                       
                        new_actions=agent.getTargetAct(state)
                        target_Q=agent.qtarget(torch.cat((new_state,new_actions),-1)).view(-1)
                        target=rewards+self.gamma*(1-done)*target_Q
                    
                    agent.optimizerQ.zero_grad()
                    Q_values=agent.q(torch.cat((state,action),-1)).view(-1)
                    Q_loss=sl(Q_values,target).mean()
                    Q_loss.backward()
                    agent.optimizerQ.step()
                    agent.Q_loss=Q_loss

                    agent.optimizerP[agent.currentPolicy].zero_grad()
                    action_pred=agent.policies[agent.currentPolicy](state)
                    actor_loss=agent.q(torch.cat((state,action_pred),-1))
                    actor_loss= -(actor_loss).mean(0)
                    actor_loss.backward()
                    agent.optimizerP[agent.currentPolicy].step()
                    agent.actor_loss=actor_loss



        for x in range(self.env.n):
            self.agents[x].soft_update()








"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    print("env ok!")
    return env #,scenario,world

def padObs(obs,size):
    return([np.concatenate((o,np.zeros(size-o.shape[0]))) if o.shape[0]<size else o for o in obs])

if __name__ == '__main__':

    env, config, outdir, logger = init('./configs/config_maddpgStart_simple_spread.yaml', "MADDPG")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]


    noise = [config["noise"](config["action_size"], sigma=config["sigma_noise"]) for i in range(env.n)]
    noiseTest = [config["noise"](config["action_size"], sigma=config["sigma_noiseTest"]) for i in range(env.n)]
    #
    #
    obs = env.reset()
    obs_n = [o.shape[0] for o in obs]
    mshape=max(obs_n)
    obs_n = [mshape for o in obs]

    agent = MADDPG(env, config, config["action_size"], obs_n, noise, noiseTest)
    #agent.load("./XP/simple_tag/MADDPG_3_pol/save_4000")
    verbose=True
    rsum = np.zeros(env.n)
    mean = np.zeros(env.n)
    itest = 0
    ntest = 0
    ntrain = 0
    rewards = np.zeros(env.n)
    done = [False]
    ended=False
    agent.nbEvents = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = np.zeros(env.n)

        obs = env.reset()
        obs = padObs(obs, mshape)
        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False
        verbose=False


        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = np.zeros(env.n)
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            #logger.direct_write("rewardTest", mean / nbTest, itest)
            for k in range(env.n):
                logger.direct_write("rewardTest/raw_" + str(k), mean[k]/nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()


        new_obs = np.array(obs)

        ended = False

        while True:

            if verbose:
                env.render(mode="none")

            obs = new_obs
            actions = agent.act(obs)

            j += 1
            new_obs, rewards, done, _ = env.step(actions)
            new_obs = padObs(new_obs, mshape)
            if ((not agent.test) and j >= int(config["maxLengthTrain"])) or (j>=int(config["maxLengthTest"])) :
                ended=True
            new_obs = np.array(new_obs)
            rewards = np.array(rewards)
            rewards_store=np.clip(rewards,-config["maxReward"],config["maxReward"])
            if (not agent.test):

                agent.store(obs, actions, new_obs, rewards_store, done,j)

                if agent.timeToLearn(ended):
                    agent.learn()

                    for k,a in enumerate(agent.agents):
                        if a.actor_loss is not None:
                            logger.direct_write("Actor_Loss/raw_"+str(k), a.actor_loss, agent.nbEvents)
                            logger.direct_write("Q_Loss/raw_"+str(k), a.Q_loss, agent.nbEvents)

            rsum += rewards

            if done[0] or ended:
                agent.endsEpisode()
                if not agent.test:
                    ntrain += 1
                    print("Train:",str(ntrain) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    for k in range(len(rsum)):
                        logger.direct_write("reward/raw_"+str(k), rsum[k], ntrain)
                
                else:
                    ntest += 1
                    print("Test:", str(ntest) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")


                mean = mean + rsum
                rsum = np.zeros(env.n)

                break


    env.close()

