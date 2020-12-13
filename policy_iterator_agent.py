import numpy as np
from amalearn.agent import AgentBase
from mdp_enviroment import MDPEnvironment
import time
import multiprocessing as mp


class PolicyIteratorAgent(AgentBase):
    def __init__(self, id, environment:MDPEnvironment):
        super(PolicyIteratorAgent, self).__init__(id, environment)
        self.environment = environment


    def take_action(self, debug_mode="True") -> (object, float, bool, object):
        pass

    def policy_eval(self, policy, theta=0.0001):


        V = np.zeros_like(self.environment.states_shape_array)
        number_of_general_actions = self.environment.num_of_actions

        # print(self.environment.calculate_reward(action=5, state_values=V))

        while True:
            delta = 0
            for index, x in np.ndenumerate(self.environment.states_shape_array):
                v = 0
                self.environment.state['current'] = index
                available_actions_list = self.environment.available_actions()
                q = np.zeros(number_of_general_actions)

                start = time.time()
                for action in np.where(policy[index[0], index[1]] != 0)[0]:

                    action = int(action)
                    q[action] = self.environment.calculate_reward(action=action, state_values=V)
                    v += policy[index[0], index[1], action] * q[action]

                end = time.time()

                delta = max(delta, np.abs(v - V[index]))
                V[index] = v

                print(index, " done "," elapsed time=", end-start)

            print(delta)

            if delta < theta:
                break

        return V

    def policy_improvement(self, theta=0.1):
        number_of_general_actions = self.environment.num_of_actions
        policy = np.zeros([self.environment.states_shape_array.shape[1], self.environment.states_shape_array.shape[0], number_of_general_actions])
        policy[:, :, self.environment.max_rental_services] = 1

        policy_to_plot = np.ones_like(self.environment.states_shape_array)


        while True:
            policy_stable = True
            V = self.policy_eval(policy, theta)
            for index, x in np.ndenumerate(self.environment.states_shape_array):
                self.environment.state['current'] = index

                available_actions_list = self.environment.available_actions()
                q = np.zeros(number_of_general_actions)


                for action in available_actions_list:
                    q[action] = self.environment.calculate_reward(action=action, state_values=V)

                best_action = np.argmax(q)

                if best_action != np.argmax(policy[index[0], index[1]]):
                    policy_stable = False

                policy[index[0], index[1]] = np.zeros(number_of_general_actions)
                policy[index[0], index[1], best_action] = 1

                policy_to_plot[index[0], index[1]] = best_action - 5

                print("improvement of index", index, " done")

            if policy_stable:
                break

        return policy, policy_to_plot, V








    # policy = np.zeros([self.environment.states_shape_array.shape[1], self.environment.states_shape_array.shape[0],
    #                    number_of_general_actions])




    def reset(self):
        available_actions = self.environment.available_actions()
        self.H = np.full(available_actions, 0, dtype=np.float64)
        self.pi = np.full(available_actions, 0, dtype=np.float64)
        self.trial_number = 0
        self.action_counter = np.full(available_actions, 0)
        self.baseline = 0

