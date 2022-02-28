
import matplotlib

from gridworld.gridworld_env import GridworldEnv
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")

import torch
import copy
import torch.nn.functional as F
from utils import *
from core import *
from memory import *



class DQN(object):
    """Deep Q-Networl with experience replay"""
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
        self.D = Memory(self.opt.mem_size)
        # Definition of Q and Q_hat
        # NN is defined in utils.py
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

        with torch.no_grad():
            self.Q_target = copy.deepcopy(self.Q)
        self.Q_target=self.Q_target.to(self.device)
        # Definition of loss
        self.Q=self.Q.to(self.device)
        self.loss = F.mse_loss

        # Optimiser
        self.lr = float(opt.lr)
        self.optim = torch.optim.Adam(self.Q.parameters(), lr=self.lr)


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

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return
        else:
            # get mini_batch a batch of (ob, action, reward, new_ob, done)
            _, _, mini_batch = self.D.sample(self.opt["mini_batch_size"])
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
            logger.direct_write("Loss", output, i)
            self.optim.zero_grad()
            output.backward()
            self.optim.step()

            if self.nbEvents % self.opt.Target_Update ==0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            

    def store(self, ob, action, new_ob, reward, done,goal, it):
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
            self.D.store(tr)
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

    env, config, outdir, logger = init('./config_curriculum.yaml', "DQN")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]
 
    # Agent
    agent = DQN(env, config)
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config)
        rsum = 0
        ob = env.reset()
        goal,_=env.sampleGoal()
        goal=agent.featureExtractor.getFeatures(goal)

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car ça ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False
        verbose=True

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
            agent.store(ob, action, new_ob, reward, done,goal, j)
            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or \
                    ((agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()                   

            if done:
                if verbose:
                    env.render()
                print("Episode %d\t rsum=%.3f\t%d actions" %(i,rsum,j))
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0
                break
    env.close()
