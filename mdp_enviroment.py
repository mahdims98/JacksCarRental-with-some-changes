import gym
from amalearn.environment import EnvironmentBase
from service_reward import ServiceRewards
import numpy as np


class MDPEnvironment(EnvironmentBase):
    def __init__(self, reward_object: ServiceRewards, episode_max_length, id, max_services, max_rental_services,
                 container=None):
        self.state_space = gym.spaces.MultiDiscrete([max_services + 1, max_services + 1])
        self.states_shape_array = np.zeros([max_services + 1, max_services + 1])
        self.num_of_states = (max_services + 1) ** 2


        for i in range(0, max_services + 1):

            for j in range(0, max_services + 1):
                self.states_shape_array[i, j] = i

                # first number: services available in A
                # second number: services available in B
        self.action_space = gym.spaces.Discrete(2 * max_rental_services + 1)
        self.num_of_actions = 2 * max_rental_services + 1
        # (2*max_rental_services+1) for max_rental_services services taken by A from B
        # 0 for max_rental_services services taken by B from A
        # max_rental_services for 0 services rented

        super(MDPEnvironment, self).__init__(self.action_space, self.state_space, id, container)
        self.max_rental_services = max_rental_services
        self.max_services = max_services
        self.reward_object = reward_object
        self.episode_max_length = episode_max_length
        self.state = {
            'current': [10, 10],
            'previous': None,
            'next': None,
            'last_action': None
        }

    def calculate_reward(self, action, state_values):
        return self.reward_object.get_reward(state_ids=self.state['current'], action=action, state_values=state_values)

    def terminated(self):
        return self.state['length'] >= self.episode_max_length

    def observe(self):
        return {}

    def available_actions(self):
        available_services_to_rent_by_b = min(self.state['current'][0], self.max_rental_services)  # services in b
        available_services_to_rent_by_a = min(self.state['current'][1], self.max_rental_services)

        available_actions = np.arange(-available_services_to_rent_by_b+self.max_rental_services, available_services_to_rent_by_a + 1+self.max_rental_services)
        return available_actions

    def next_state(self, action):
        self.episode_length += 1

        self.state['previous'] = self.state['current']
        self.state['current'][0] = self.state['previous'][0] + action
        self.state['current'][1] = self.state['previous'][1] - action
        self.state['last_action'] = action

    def reset(self):
        self.state = {
            'current': [10, 10],
            'previous': None,
            'last_action': None
        }

    def render(self, mode='human'):
        print('{}:\taction={}'.format(self.state['length'], self.state['last_action']))

    def close(self):
        return
