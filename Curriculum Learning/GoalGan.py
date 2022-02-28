import matplotlib
from gridworld.gridworld_env import GridworldEnv
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import random
import torch
import copy
import torch.nn.functional as F
from utils import *
from core import *
from memory import *
from torch.distributions import Uniform, Normal
import numpy as np


class GoalGan(object):

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
        
        # Definition of Q and Q_hat
        # NN is defined in utils.py
        state_feature_size = self.featureExtractor.outSize
        action_feature_size = self.action_space.n
        
        self.device=torch.device("cuda")
    
        self.latent_dim=4
        self.p_z=Uniform(-torch.ones(self.latent_dim),torch.ones(self.latent_dim))

        self.disc=nn.Sequential(
            nn.Linear(state_feature_size,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,1)
        )
        self.disc=self.disc.to(self.device)
        self.gen=nn.Sequential(
            nn.Linear(self.latent_dim,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,state_feature_size)
        )
        self.gen=self.gen.to(self.device)
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



        self.goal_memory=[]
        self.eps=3.
        
        self.Rmin=-9.
        self.Rmax=-0.5

        self.batch_size=self.opt["mini_batch_size"]

        # Optimiser
        self.lr = float(opt.lr)
        self.optim_q = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.optim_g = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        self.optim_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr)




    def sample_goals(self):
        noise=self.p_z.sample((8,)).to(self.device)
        with torch.no_grad():
            gen_goals=self.gen(noise)
            gen_goals=gen_goals.type(torch.int64)
        gen_goals=gen_goals.cpu().numpy().reshape(8,2)
        

        idx_samples=random.sample(range(len(self.goal_memory)),min(4,len(self.goal_memory)))

        goals=[]
        for i in range(gen_goals.shape[0]):
            g_exist=any((x[0]==gen_goals[i]).all() for x in goals)
            if not g_exist:
                goals.append([gen_goals[i],0,0,None])
        for idx in idx_samples:
            g_exist=any((x[0]==self.goal_memory[idx][0]).all() for x in goals)
            if not g_exist:
                goals.append( [self.goal_memory[idx][0],self.goal_memory[idx][1],self.goal_memory[idx][2],idx] ) 

        return goals
        

    def label_goals(self,goal_tuples):
        labels_train=np.zeros(len(goal_tuples))
        goals_train=np.zeros((len(goal_tuples),2))

        for (j,(g,n,r,_)) in enumerate(goal_tuples):
            goals_train[j,:]=g
            if n!=0:
                mean=r/n
                if mean>=self.Rmin and mean<=self.Rmax:
                    labels_train[j]=1
            else:
                mean=0
                
        goals_train=torch.cuda.FloatTensor(goals_train,device=self.device).view(-1,2)
        labels_train=torch.cuda.FloatTensor(labels_train,device=self.device).view(-1)

        return goals_train,labels_train

    def update_goals(self,goal_tuples):
        
        for (g,n,r,i) in goal_tuples:
            if i is not None:
                self.goal_memory[i][1]+=n
                self.goal_memory[i][2]+=r

            else:
                if n!=0:
                    if r/n>=self.Rmin and r/n<=self.Rmax:
                        self.add_goal(g.reshape(-1),n,r,self.eps)



    def add_goal(self, g,n,r,eps):
        
        goals_mem=np.stack(list(zip(*self.goal_memory))[0])
        delta_dist = goals_mem - g
        distance = min(np.sqrt((delta_dist[:,0])**2+(delta_dist[:,1])**2))

        if distance>eps:
            self.goal_memory.append( [g,n,r] )



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

    # sauvegarde du modèle
    def save(self, outputDir):
        # torch.save
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass


    def learn_DQN(self):
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
            output = self.loss(y_batch, q_batch)
            #logger.direct_write("Loss", output, i)
            self.optim_q.zero_grad()
            output.backward()
            self.optim_q.step()

            if self.nbEvents % self.opt.Target_Update ==0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            

            


    def learn_gan(self,goals,labels):
        
        self.optim_d.zero_grad()

        real_pred=self.disc(goals).view(-1)
        lat_samples=self.p_z.sample((real_pred.shape[0],)).to(self.device)
        with torch.no_grad():
            gen_goals=self.gen(lat_samples)
        gen_pred=self.disc(gen_goals).view(-1)
        loss_d= labels*torch.pow(real_pred-torch.ones_like(real_pred),2) + \
                (1-labels)*torch.pow(real_pred+torch.ones_like(real_pred),2)+ \
                torch.pow(gen_pred,2)
        loss_d=loss_d.mean()
        loss_d.backward()
        self.optim_d.step()


        self.optim_g.zero_grad()
        lat_samples=self.p_z.sample((real_pred.shape[0],)).to(self.device)
        gen_goals=self.gen(lat_samples)
        gen_pred=self.disc(gen_goals).view(-1)
        loss_g=F.mse_loss(gen_pred,torch.zeros_like(gen_pred))
        loss_g.backward()
        self.optim_g.step()

        




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

    def timeToLearn(self, done):
        # retoune vrai si c'est le moment d'entraîner l'agent.
        # Dans cette version retourne vrai tous les freqoptim evenements
        # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
        if self.test:
            return False
        self.nbEvents += 1
        return self.nbEvents % self.opt.freqOptim == 0








if __name__ == '__main__':
    # Configuration

    env, config, outdir, logger = init('./config_curriculum.yaml', "GoalGan")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]
    warm_up = config["warm_up"]
    roll_out = config["roll_out"]
 
    # Agent
    agent = GoalGan(env, config)
    rsum = 0
    mean = 0
    verbose = False
    itest = 0
    reward = 0
    done = False
    goal,_=env.sampleGoal()
    goal=agent.featureExtractor.getFeatures(goal)

    agent.goal_memory.append( [goal.reshape(-1),0,0] )


    for i in range(warm_up):
        idx=np.random.randint(0,len(agent.goal_memory),1)[0]
        goal=agent.goal_memory[idx][0]
        rsum = 0
        ob = env.reset()

        
        new_ob = agent.featureExtractor.getFeatures(ob)
        j=0
        while True:
            
            ob = new_ob
            action = agent.act(ob,goal)

            new_ob, _, _, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)
            done=(new_ob==goal).all()
            j += 1

            if done or j == config["maxLengthTrain"]:
                agent.add_goal(new_ob.reshape(-1),0,0,0)
                break
        

    for big_iter in range(episode_count):

        goal_tuples=agent.sample_goals()

        for i in range(roll_out):

            rsum = 0
            ob = env.reset()
            samp_idx=np.random.randint(0,len(goal_tuples),1)[0]
            goal=goal_tuples[samp_idx][0]

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
                agent.test = True

            # On a fini cette session de test
            if i % freqTest == nbTest:
                print("End of test, mean reward=", mean / nbTest)
                itest += 1
                logger.direct_write("rewardTest", mean / nbTest, itest)
                agent.test = False

            # C'est le moment de sauver le modèle
            if i % freqSave == 0:
                agent.save(outdir + "/save_" + str(i))

            j = 0  # steps in an episode
            if verbose:
                env.render()

            new_ob = agent.featureExtractor.getFeatures(ob)
            while True:
                if verbose:
                    env.render()

                ob = new_ob
                action = agent.act(ob,goal)

                new_ob, _, _, _ = env.step(action)
                new_ob = agent.featureExtractor.getFeatures(new_ob)
                done=(new_ob==goal).all()
                reward=1.0 if done else -0.1
                agent.store(ob, action, new_ob, reward, done,goal)
                j += 1

                # Si on a atteint la longueur max définie dans le fichier de config
                if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or \
                        ((agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                    done = True
                    print("forced done!")

                
                rsum += reward

                if agent.timeToLearn(done):
                    agent.learn_DQN()                   

                if done:
                    if verbose:
                        env.render()
                    print("Episode %d\t rsum=%.3f\t%d actions" %(i,rsum,j))
                    logger.direct_write("reward", rsum, i)
                    mean += rsum
                    

                    if agent.test:
                        goal_tuples[samp_idx][1]+=1
                        goal_tuples[samp_idx][2]+=rsum

                    rsum = 0

                    break


        goals_train,labels_train=agent.label_goals(goal_tuples)

        for _ in range(200):
            agent.learn_gan(goals_train,labels_train)

        agent.update_goals(goal_tuples)

    env.close()
