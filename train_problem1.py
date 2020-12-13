import numpy as np
import seaborn as sns


from policy_iterator_agent import PolicyIteratorAgent
from service_reward import ServiceRewards
from mdp_enviroment import MDPEnvironment

import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


lambdas = {'a_demand':3, 'a_cancellation':3, 'b_demand':4,  'b_cancellation':2}
reward_object = ServiceRewards(lambdas=lambdas, service_reward=10, rent_punishment=-2, no_service_punishment=-5, maximum_services=20,maximum_rent=5, poisson_upper_bound=12, discount_factor=0.9)
env = MDPEnvironment(reward_object=reward_object, episode_max_length=100, id=1, max_services=20, max_rental_services=5)
policy_iterator_agent = PolicyIteratorAgent('1', env)

# policy = np.zeros([self.environment.states_shape_array.shape[1], self.environment.states_shape_array.shape[0],
#                    number_of_general_actions])
start = time.time()
policy, policy_to_plot, value = policy_iterator_agent.policy_improvement(theta=0.1)
end = time.time()
print((end - start)/60)

print("State Values:\n", value)
print("Optimal Policy:\n", policy)


plot1 = plt.imshow(policy_to_plot, cmap="YlGnBu", interpolation='nearest')

c_bar = plt.colorbar(plot1)
plt.show()


x = range(policy_to_plot.shape[0])
y = range(policy_to_plot.shape[0])
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.scatter(X, Y, policy_to_plot)

plt.show()


plot1 = plt.imshow(value, cmap="YlGnBu", interpolation='nearest')
c_bar = plt.colorbar(plot1)
plt.show()


x = range(policy_to_plot.shape[0])
y = range(policy_to_plot.shape[0])
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.scatter(X, Y, value)

plt.show()

np.save("values.npy", value)
np.save("policy_to_plot.npy", policy_to_plot)
np.save("policy.npy", policy)











