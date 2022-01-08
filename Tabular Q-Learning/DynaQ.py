import copy
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


class DynaQ(object):
    def __init__(self, env, opt):
        self.opt = opt
        self.action_space = env.action_space
        self.env = env
        self.discount = opt.gamma
        self.alpha = opt.learningRate
        self.alpha_r = opt.rewardLearningRate
        self.alpha_p = opt.MDPLearningRate
        self.explo = opt.explo
        self.exploMode = opt.exploMode  # 0: epsilon greedy, 1: ucb
        self.eps = opt.explo
        self.decay = opt.decay
        self.test = False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles
        self.model = {}  # a model of three degree dictionary, store (s, a, s' , p)
        self.reward = {}  # a model of three degree dictionary, store (s, a, s' , r)
        self.K = opt.nbModelSamples
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

    def store_model(self, s, a, s_prime, r):
        self.model.update({s: {a: {s_prime: 1}}})
        self.reward.update({s: {a: {s_prime: r}}})

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
            self.values[self.s][self.a] = \
                self.values[self.s][self.a] + \
                self.alpha * (self.r - self.values[self.s][self.a])
        else:
            self.values[self.s][self.a] = \
                self.values[self.s][self.a] + \
                self.alpha * (self.r + self.discount * self.values[self.s_p].max() -
                              self.values[self.s][self.a])
        # learn for model
        self.reward[self.s][self.a][self.s_p] =\
            self.reward[self.s][self.a][self.s_p] + self.alpha_r * (self.r - self.reward[self.s][self.a][self.s_p])
        for state in self.model[self.s][self.a]:
            if state == self.s_p:
                self.model[self.s][self.a][state] = \
                    self.model[self.s][self.a][state] + self.alpha_p * (1 - self.model[self.s][self.a][state])
            else:
                self.model[self.s][self.a][state] = \
                    self.model[self.s][self.a][state] + self.alpha_p * (0 - self.model[self.s][self.a][state])
        self.learn_from_model(self.K)

    def learn_from_model(self, k):
        """Update Q from dynamic model, where we sample K state-action couples"""
        for _ in range(k):
            rand_idx = np.random.choice(range(len(self.model.keys())))
            _s = list(self.model)[rand_idx]
            rand_idx = np.random.choice(range(len(self.model[_s].keys())))
            _a = list(self.model[_s])[rand_idx]

            y = 0
            for _s_p in self.model[_s][_a]:
                 y += self.model[_s][_a][_s_p] * (self.reward[_s][_a][_s_p] + self.discount * self.values[_s_p].max())

            self.values[_s][_a] = self.values[_s][_a] + self.alpha * (y - self.values[_s][_a])


if __name__ == '__main__':
    algoName = 'DynaQ-plan5'
    env, config, outdir, logger = init('./configs/config_dynaQ_gridworld.yaml', algoName)  # in util.py
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = DynaQ(env, config)

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
            agent.store_model(ob, action, new_ob, reward)

            j += 1
            if (config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"]):
                done = True

            agent.store(ob, action, new_ob, reward, done, j)
            if not agent.test:
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
