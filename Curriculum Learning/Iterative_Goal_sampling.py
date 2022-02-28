import matplotlib
from gridworld.gridworld_env import GridworldEnv
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import random
import torch
import copy
import torch.nn.functional as F
from utils import *
from core import *
from memory import *
from torch.distributions import Uniform, Normal
import numpy as np
from collections import deque


class Goal_Sampl(object):

    def __init__(self, env, opt):
        self.opt = opt
        print(opt)
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test = False
        self.nbEvents = 0
        self.discount = opt.discount
        self.decay = opt.decay
        self.eps = opt.eps

        # Definition of replay memory D
        self.memory = Memory(self.opt.mem_size)
        
        state_feature_size = self.featureExtractor.outSize
        action_feature_size = self.action_space.n
        
        self.device=torch.device("cuda")
    
        self.Q=nn.Sequential(
            nn.Linear(2*state_feature_size,200),
            nn.Tanh(),
            nn.Linear(200,200),
            nn.Tanh(),
            nn.Linear(200,action_feature_size)
        ) 
        self.Q=self.Q.to(self.device)
        with torch.no_grad():
            self.Q_target = copy.deepcopy(self.Q)
        self.Q_target=self.Q_target.to(self.device)
        self.loss=F.mse_loss

        self.goal_memory = deque(maxlen=opt.length_goal_mem) ##FIFO list, Each element has form of [goal,n,v,H]
        self.beta=opt.beta
        self.alpha=opt.alpha
        self.buff_count=0
        self.train_count=0
    
        self.batch_size=self.opt["mini_batch_size"]

        # Optimiser
        self.lr = float(opt.lr)
        self.optim_q = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
    

    def update_goals(self,idx,done):

        if not self.test:
            self.goal_memory[idx][1]+=1
            if done:
                self.goal_memory[idx][2]+=1
            (_,n,v,_)=self.goal_memory[idx]  
            self.goal_memory[idx][3]=-v/n*np.log(max(v/n,1e-6))-(1-v/n)*np.log(max(1-v/n,1e-6))


    def sample_goal(self):
        if self.test:
            goal,_=self.env.sampleGoal()
            goal=agent.featureExtractor.getFeatures(goal)
            idx=None
            b=0
        else:
            if len(self.goal_memory)==0:
                goal,_=self.env.sampleGoal()
                goal=agent.featureExtractor.getFeatures(goal)
                idx=None
                b=0
            else:
                b=np.random.binomial(size=1, n=1, p= self.beta)
                if 1-b:
                    goal,_=self.env.sampleGoal()
                    goal=agent.featureExtractor.getFeatures(goal)
                    idx=None
                elif b:
                    entropies=np.array(list(zip(*self.goal_memory))[3])
                    probs=np.exp(self.alpha*entropies)
                    norm_factor=probs.sum()
                    probs/=norm_factor 
                    idx=np.random.choice(np.arange(len(self.goal_memory)),size=1,p=probs)[0]
                    goal=self.goal_memory[idx][0]

        return goal,b,idx

   

    def add_goal(self,g):

        if len(self.goal_memory)==0:
            self.goal_memory.append([g,1,1,0])
        else:
            goals_mem=np.stack(list(zip(*self.goal_memory))[0]).reshape(-1,2)
            delta_dist = (goals_mem - g)
            distance = min(np.sqrt((delta_dist[:,0])**2+(delta_dist[:,1])**2))

            if distance>self.opt.distance_goals:
                self.buff_count=0
                self.goal_memory.append([g,1,1,0])
           

    def timeToFeed(self):
        self.buff_count+=1
        if self.buff_count>=self.opt.freqBuffer and not self.test:
            return True
        else:
            return False
        

    def act(self, obs,goal):
        # epsilon greedy action
        if self.test:
            obs=torch.Tensor(obs).to(self.device).view(-1)
            goal=torch.Tensor(goal).to(self.device).view(-1)
            obs_goal=torch.cat((obs,goal))
            a = torch.argmax(self.Q(obs_goal)).item()
        else:
            self.eps = self.eps * self.decay
            if torch.rand(1) < self.eps:
                a = self.action_space.sample()
            else:
                obs=torch.Tensor(obs).to(self.device).view(-1)
                goal=torch.Tensor(goal).to(self.device).view(-1)
                obs_goal=torch.cat((obs,goal))
                a = torch.argmax(self.Q(obs_goal)).item()
        return a


    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return
        else:
            # get mini_batch a batch of (ob, action, reward, new_ob, done)
            _, _, mini_batch = self.memory.sample(self.batch_size)
            column_mini_batch = list(zip(*mini_batch))
            obs_batch = torch.cuda.FloatTensor(column_mini_batch[0],device=self.device).view(self.opt["mini_batch_size"],-1)# B, dim_obs=4
            action_batch = torch.cuda.LongTensor(column_mini_batch[1],device=self.device).view(self.opt["mini_batch_size"],-1)
            r_batch = torch.cuda.FloatTensor(column_mini_batch[2],device=self.device).view(-1)
            new_obs_batch = torch.cuda.FloatTensor(column_mini_batch[3],device=self.device).view(self.opt["mini_batch_size"],-1) # B, dim_obs=4
            done_batch = torch.cuda.FloatTensor(column_mini_batch[4],device=self.device).view(-1)
            goal_batch=torch.cuda.FloatTensor(column_mini_batch[5],device=self.device).view(self.opt["mini_batch_size"],-1)

            state_goal=torch.cat((obs_batch,goal_batch),dim=-1)
            new_state_goal=torch.cat((new_obs_batch,goal_batch),dim=-1)
            
            with torch.no_grad():
                Q_max=torch.max(self.Q_target(new_state_goal), axis=-1).values.view(-1) 
                y_batch = r_batch + self.discount * Q_max* (1 - done_batch) # if done this term is 0

            q_batch = self.Q(state_goal).gather(1, action_batch).view(-1) # reward self.Q.forward(obs_batch): B, 2
            q_loss = self.loss(y_batch, q_batch)
            self.optim_q.zero_grad()
            q_loss.backward()
            self.optim_q.step()
            self.q_loss=q_loss.item()
            self.train_count+=1

            if self.train_count % self.opt.Target_Update==0:
                self.Q_target.load_state_dict(self.Q.state_dict())



    def store(self, ob, action, new_ob, reward, done,goal):
        """enregistrement de la transition pour exploitation par learn ulterieure"""
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentis
            # alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            """  if it == self.opt.maxLengthTrain:
                print("undone")
                done = False """
            done=float(done)
            tr = (ob, action, reward, new_ob, done,goal)
            self.memory.store(tr)
            # ici on n'enregistre que la derniere transition pour traitement immédiat,
            # mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.lastTransition = tr

    def timeToLearn(self):
        # retoune vrai si c'est le moment d'entraîner l'agent.
        # Dans cette version retourne vrai tous les freqoptim evenements
        # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
        if self.test:
            return False
        self.nbEvents += 1
        return self.nbEvents % self.opt.freqOptim == 0 


    # sauvegarde du modèle
    def save(self, outputDir):
        # torch.save
        torch.save(self.Q.state_dict(),outputDir)

    # chargement du modèle.
    def load(self, inputDir):
        self.Q.load_state_dict(torch.load(inputDir))





if __name__ == '__main__':
    # Configuration

    env, config, outdir, logger = init('./config_curriculum.yaml', "Goal_Sampl")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]
    
    # Agent
    agent = Goal_Sampl(env, config)
    #agent.load("./XP/gridworld-v0/It_Goal_Sampl/save_3000.pth")

    rsum = 0
    mean = 0
    mean_done=0 
    verbose = False
    itest = 0
    final_goal=agent.featureExtractor.getFeatures(env.sampleGoal()[0])
    
    for i in range(episode_count):
                
        # On souhaite afficher l'environnement (attention à ne pas trop afficher car ça ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False
        verbose=False

        # C'est le moment de tester l'agent
        if i % freqTest == 0:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            mean_done=0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("Rewards/Test", mean / nbTest, itest)
            logger.direct_write("Done/Test", mean_done / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0 and i>0:
            agent.save(outdir + "/save_" + str(i)+".pth")

        j=0
        goal=None
        episode_exp=[]
        ob = env.reset()
        new_ob = agent.featureExtractor.getFeatures(ob)

        if verbose:
            env.render()

        while j<config["maxLengthTrain"]:

            if goal is None:
                goal,b,idx=agent.sample_goal()

            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob,goal)

            new_ob, _, _, _ = env.step(action)

            new_ob = agent.featureExtractor.getFeatures(new_ob)
            done=(new_ob==goal).all()
            rew=1 if done else -0.1
            rsum+=rew

            episode_exp.append((ob,action,new_ob))

            j += 1
                   
            if done or j==config["maxLengthTrain"]:
                if agent.test:
                    break
                else:
                    for (old,act,new) in episode_exp:
                        d=(new==goal).all()
                        r=1.0 if d else -0.1
                        agent.store(old,act,new,r,d,goal)

                    her_goal=new_ob
                    for (old,act,new) in episode_exp:
                        d=(new==her_goal).all()
                        r=1.0 if d else -0.1
                        agent.store(old,act,new,r,d,her_goal)

                    if b:
                        agent.update_goals(idx,done)

                    goal=None    

        if agent.timeToFeed():
            agent.add_goal(new_ob) 

        if agent.timeToLearn():
            for _ in range(agent.opt.nbOptim):
                agent.learn()
                logger.direct_write("Train Loss",agent.q_loss, agent.train_count)

        if verbose:
            env.render()

        print("Episode %d\t rsum=%.3f\t%d actions\t x-coord %d\t y-coord %d" %(i,rsum,j,new_ob[0][1],new_ob[0][0]))
        logger.direct_write("Rewards/Overall", rsum, i)
        logger.direct_write("Coordinates/x", new_ob[0][1], i)
        logger.direct_write("Coordinates/y", new_ob[0][0], i)
        if new_ob[0][1]==4 and new_ob[0][0]==1:
            print("---DONE---")
            logger.direct_write("Done/Overall",1,i)
            mean_done+=1
        else:
            logger.direct_write("Done/Overall",0,i)
        mean+=rsum
        rsum=0



    env.close()
