import gym
import gridworld

import numpy as np
from numpy import linalg as LA


class ValueIterationAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, e=1e-6, gamma=0.99):
        self.env = env
        self.states, self.mdp = env.getMDP()
        self.e = e
        self.gamma = gamma
        self.len_states = len(self.states)
        self.PI = np.zeros(self.len_states)  # initialise PI

    def train(self):
        i = 0
        # v the optimal evaluation function to be converged
        v = np.zeros(self.len_states)
        while True:
            new_v = np.zeros(self.len_states)
            for state, A_table in self.mdp.items():
                values_a = {}
                for action,  s_prime_list in A_table.items():
                    value = 0
                    for p, s_prime, r, done in s_prime_list:
                        value += p * (r + self.gamma * v[s_prime])
                    values_a[action] = value
                new_v[state] = max(values_a.values())
            if LA.norm(v - new_v) < self.e:
                print("value function converged!")
                print(f"iteration: {i}")
                # print(f"value: {v}")
                break
            v = new_v
            i += 1

        for state, A_table in self.mdp.items():
            values_a = {}
            for action, s_prime_list in A_table.items():
                value = 0
                for p, s_prime, r, done in s_prime_list:
                    value += p * (r + self.gamma * v[s_prime])
                values_a[action] = value
            self.PI[state] = max(values_a, key=values_a.get)

        # print(f"policy: {self.PI}")
        self.display_policy()

    def display_policy(self):
        obs = self.env.reset()
        rsum = 0
        j = 0
        while True:
            action = self.act(obs)
            obs, reward, done, _ = self.env.step(action)
            rsum += reward
            j += 1
            self.env.render()
            if done or j > 1000:
                print("rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    def act(self, observation):
        observed_state = env.getStateFromObs(observation)
        return self.PI[observed_state]


if __name__ == '__main__':
    env = gym.make("gridworld-v0")
    # env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -2, 6: -1})
    # env.setPlan("gridworldPlans/plan1.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan2.txt", {0: -0.001, 3: 1, 4: 0.05, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan3.txt", {0: -0.00001, 3: 1, 4: 1, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan4.txt", {0: -1, 3: 1, 4: 0, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan5.txt", {0: -0.001, 3: 1, 4: 0.1, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan6.txt", {0: -0.001, 3: 4, 4: 0.1, 5: -1, 6: -0.1})
    # env.setPlan("gridworldPlans/plan7.txt", {0: -0.1, 3: 0.1, 4: 1, 5: -0.1, 6: -0.1})
    # env.setPlan("gridworldPlans/plan8.txt", {0: -1, 3: 5, 4: 1, 5: -1, 6: -1})
    env.setPlan("gridworldPlans/plan9.txt", {0: -0.001, 3: 1, 4: 2, 5: -1, 6: -0.1}) # trop lourd
    # env.setPlan("gridworldPlans/plan10.txt", {0: -0.001, 3: 1, 4: 1, 5: -0.0001, 6: -1})
    env.seed(0)  # Initialise le seed du pseudo-random
    # print(env.action_space)  # Quelles sont les actions possibles
    # print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    # env.render(mode="human")  # visualisation sur la console
    # states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats
    # print("Nombre d'etats : ", len(states))
    # state, transitions = list(mdp.items())[0]
    # print(state)  # un etat du mdp
    # print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    ValueIterationAgent = ValueIterationAgent(env, gamma=0.9)
    ValueIterationAgent.train()
