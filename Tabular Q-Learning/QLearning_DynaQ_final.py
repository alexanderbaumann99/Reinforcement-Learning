import matplotlib
from numpy import random
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from datetime import datetime
import os
from utils import *


class QLearning(object):


    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.explo=opt.explo
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.sarsa=opt.sarsa
        self.modelSamples=opt.nbModelSamples
        self.model = {}
        self.probas = {}
        self.test=False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles
        self.state_actions = []



    def save(self,file):
       pass


    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self,obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            self.values.append(np.ones(self.action_space.n) * 1.0) # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
        return ss

    def act(self, obs):
        mx_nx_reward = -999
        eps = self.explo
        p = np.random.random()
        if p < eps:
            action = np.random.choice(4)
        else:
            for a in range(4):
                nxt_reward = self.values[obs][a]
                if nxt_reward >= mx_nx_reward:
                    action = a
                    mx_nx_reward = nxt_reward
        return(action)
    # def act(self, obs):
    #     eps = self.explo
    #     if random.random() < eps:
    #         a = self.action_space.sample()
    #     else:
    #         possible_actions = self.values[obs]
    #         a = possible_actions.argmax()
    #     return a


    def store(self, ob, action, new_ob, reward, done, it):
        if self.test:
            return
        self.last_source=ob
        self.last_action=action
        self.last_dest=new_ob
        self.last_reward=reward
        if it == self.opt.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done=done
        self.explo = np.maximum(self.explo-0.001*it, 0)

    def learn(self, done):
        dynaq = True # change to false for pure QLearning
        alphaR = 0.1
        if done:
            # pas de max Q, juste r_t - Q(s_t)
            pass
        if not done:
            self.state = self.last_source
            action = self.last_action
            self.state_actions.append((self.state, action))

            new_state = self.last_dest
            reward = self.last_reward

            q = self.values[self.state][action]
            alpha = self.alpha
            gamma = self.discount
            if self.sarsa:
                self.values[self.state][action] = q + alpha*(reward + gamma*(self.values[new_state][action])-q)
            else:
                self.values[self.state][action] += alpha*(reward + gamma*np.max(self.values[new_state])-self.values[self.state][action])
                if dynaq:
                    if self.state not in self.model.keys():
                        self.model[self.state] = {}
                        self.probas[self.state] = {}
                    if action not in self.model[self.state]:
                        self.model[self.state][action] = {new_state: reward}
                        self.probas[self.state][action] = {new_state: 1}
                    if new_state not in self.model[self.state][action]:
                        self.model[self.state][action][new_state] = reward
                        self.probas[self.state][action][new_state] = 1
                    else:                     
                        self.model[self.state][action][new_state] += alphaR*(reward-self.model[self.state][action][new_state])
                        self.probas[self.state][action][new_state] += alphaR*(1-self.probas[self.state][action][new_state])
                        for other_state in self.model[self.state][action]:
                            self.probas[self.state][action][other_state] += alphaR*(1-self.probas[self.state][action][other_state])
                    self.state = new_state

                    for _ in  range(100):
                        rand_idx = np.random.choice(range(len(self.model.keys())))
                        _state = list(self.model)[rand_idx]
                        rand_idx = np.random.choice(range(len(self.model[_state].keys())))
                        _action = list(self.model[_state])[rand_idx]
                        for other_state in self.model[_state][_action]:
                            self.values[_state][_action] += alpha*(self.probas[_state][_action][other_state]*(self.model[_state][_action][other_state] + gamma*np.max(list(self.values[other_state]))) - self.values[_state][_action])

        else:
            pass


if __name__ == '__main__':
    algoName = 'DynaQ-plan9'
    env,config,outdir,logger=init('configs/config_dynaQ_neomi_gridworld.yaml', algoName)

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = QLearning(env, config)
    

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)  # permet de changer la config en cours de run

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()
        if (i > 0 and i % int(config["freqVerbose"]) == 0):
            verbose = True
        else:
            verbose = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()
        new_ob = agent.storeState(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.storeState(new_ob)

            j+=1

            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                #print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            agent.learn(done)
            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write(f"rewardTrain/{config['map']}", rsum, i)
                mean += rsum
                break



    env.close()