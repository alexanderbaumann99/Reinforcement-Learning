import time

import gym
import gridworld

import numpy as np
from numpy import linalg as LA


class PolicyIterationAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, e=1e-6, gamma=0.99, display_frequence=50):
        self.env = env
        self.states, self.mdp = env.getMDP()
        self.e = e
        self.gamma = gamma
        self.len_states = len(self.states)
        self.PI = np.zeros(self.len_states)  # initialise PI
        self.display_frequence = display_frequence

    def train(self):
        i = 0
        total_i = 0
        while True:
            # v_pi evaluation function for policy self.PI
            v_pi = np.zeros(self.len_states)
            while True:
                new_v_pi = np.zeros(self.len_states)
                for state, A_table in self.mdp.items():
                    # for each state in v
                    action = self.PI[state]
                    s_prime_list = A_table[action]
                    for p, s_prime, r, done in s_prime_list:
                        # for each s_prime possible in state s with action PI[state]
                        new_v_pi[state] += p * (r + self.gamma * v_pi[s_prime])
                if LA.norm(v_pi - new_v_pi) < self.e:
                    break
                total_i += 1
                v_pi = new_v_pi

            # Policy update for each state
            new_pi = np.zeros_like(self.PI)
            for state, A_table in self.mdp.items():
                values_a = {}
                for action,  s_prime_list in A_table.items():
                    value = 0
                    for p, s_prime, r, done in s_prime_list:
                        value += p * (r + self.gamma * v_pi[s_prime])
                    values_a[action] = value
                new_pi[state] = max(values_a, key=values_a.get)
            # if i % self.display_frequence == 0:
            #     print(f"policy iteration: {i}")
            #     # print(f"policy: {self.PI}")
            #     # print(f"value: {v_pi}")
            #     self.display_policy()
            if np.array_equal(new_pi, self.PI):
                print("Policy converged!")
                # print(f"new policy: {new_pi}")
                # print(f"policy: {self.PI}")
                print(f"Total iteration: {total_i}")
                self.display_policy()
                break
            self.PI = new_pi
            i += 1

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
            if done or j > 100:
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
    # env.setPlan("gridworldPlans/plan3.txt", {0: 0, 3: 1, 4: 1, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan4.txt", {0: -0.001, 3: 2, 4: 100, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan5.txt", {0: -0.001, 3: 1, 4: 2, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan6.txt", {0: -0.001, 3: 1, 4: 2, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan7.txt", {0: -0.001, 3: 1, 4: 2, 5: -1, 6: -0.1})
    # env.setPlan("gridworldPlans/plan8.txt", {0: -1, 3: 5, 4: 1, 5: -1, 6: -1})
    # env.setPlan("gridworldPlans/plan9.txt", {0: -0.001, 3: 1, 4: 2, 5: -1, 6: -0.1}) # trop lourd
    # env.setPlan("gridworldPlans/plan10.txt", {0: -0.001, 3: 2, 4: 1, 5: -1, 6: -1})
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
    PolicyIterationAgent = PolicyIterationAgent(env, gamma=0.99)
    PolicyIterationAgent.train()
    time.sleep(5)

