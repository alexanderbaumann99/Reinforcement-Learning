# Q lambda with eligibility trace

from datetime import datetime
import os
import random

import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np

from utils import *

matplotlib.use("TkAgg")


class QLambda(object):
    def __init__(self, env, opt):
        self.opt = opt
        self.action_space = env.action_space
        self.env = env
        self.discount = opt.gamma
        self.alpha = opt.learningRate

        self.explo = opt.explo
        self.exploMode = opt.exploMode  # 0: epsilon greedy, 1: ucb
        self.eps = opt.explo
        self.decay = opt.decay
        self.test = False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles

        # eligibity
        self.e = {}  # dictionary of eligibility (s, a): e
        self.Lambda = opt.eligibility  # Q(lambda) learning
        self.threshold = opt.eligibilityThreshold

    def save(self, file):
        pass

    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self, obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)  # if haven't met, return -1

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
            self.values.append(np.ones(self.action_space.n) * 1.0)
        return ss

    def act(self, obs):
        self.eps = self.eps * self.decay
        if random.random() < self.eps:
            a = self.action_space.sample()
        else:
            possible_actions = self.values[obs]
            a = possible_actions.argmax()
        return a

    def store(self, ob, action, new_ob, reward, done, it):
        if self.test:
            return
        self.s = ob
        self.a = action
        self.s_p = new_ob
        self.r = reward
        if it == self.opt.maxLengthTrain:
            # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done = done

    def learn(self, done):
        # learn for Q
        if done:
            delta = self.r - self.values[self.s][self.a]
        else:
            delta = self.r + self.discount * self.values[self.s_p].max() - self.values[self.s][self.a]
        for s, a in self.e:
            self.values[s][a] = self.values[s][a] + self.alpha * delta * self.e[(s, a)]
            self.e[(s, a)] = self.Lambda * self.discount * self.e[(s, a)]


if __name__ == '__main__':
    algoName = 'QLambda-plan9'
    env, config, outdir, logger = init('./configs/config_qlambda_gridworld.yaml', algoName)  # in util.py
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = QLambda(env, config)

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
        ob = env.reset()
        if i > 0 and i % int(config["freqVerbose"]) == 0:
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
            if (ob, action) in agent.e.keys():
                agent.e[(ob, action)] += 1
            else:
                agent.e.update({(ob, action): 1})

            j += 1
            if (config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"]):
                done = True

            agent.store(ob, action, new_ob, reward, done, j)
            agent.learn(done)
            rsum += reward
            if done:
                # tensoboard logging
                if verbose:
                    env.render()
                print(algoName)

                logger.direct_write(f"rewardTrain/{config['map']}", rsum, i)

                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                break
    env.close()
