from amalearn.reward import RewardBase
import numpy as np
from poisson_calculator import PoissonCalculator
import cProfile
import pstats



class ServiceRewards(RewardBase):
    def __init__(self, lambdas: dict, service_reward, rent_punishment, no_service_punishment, maximum_services,
                 maximum_rent, poisson_upper_bound=13, discount_factor=1):
        """
        """
        super(ServiceRewards, self).__init__()
        # self.labmda_a_demand = lambdas['a_demand']
        # self.labmda_b_demand = lambdas['b_demand']
        # self.labmda_a_cancellation = lambdas['a_cancellation']
        # self.labmda_b_cancellation = lambdas['b_cancellation']

        self.lambdas = lambdas
        self.maximum_services = maximum_services
        self.maximum_rent = maximum_rent
        self.service_reward = service_reward
        self.rent_punishment = rent_punishment  # must be negative
        self.no_service_punishment = no_service_punishment
        self.poisson_upper_bound = poisson_upper_bound
        self.poisson_calc_a_demand = PoissonCalculator(lam=self.lambdas['a_demand'])
        self.poisson_calc_b_demand = PoissonCalculator(lam=self.lambdas['b_demand'])
        self.poisson_calc_a_cancellation = PoissonCalculator(lam=self.lambdas['a_cancellation'])
        self.poisson_calc_b_cancellation = PoissonCalculator(lam=self.lambdas['b_cancellation'])
        self.discount_factor = discount_factor

    def get_reward(self, state_ids, action, state_values):
        current_state_id_a = state_ids[0]
        current_state_id_b = state_ids[1]
        action = action - self.maximum_rent # between -5 and 5

        assert np.abs(action) <= self.maximum_rent, "action is not valid "
        reward = 0
        expected_reward = 0
        # input action range is 0 to 2*max_rent_services+1. we map it to -max_rent_services to max_rent_services so:
        # max_rent_services for max_rental_services services taken by A from B
        # -max_rent_services for max_rental_services services taken by B from A
        # 0 for 0 services rented

        a_new_capacity = min(current_state_id_a + action, self.maximum_services)
        b_new_capacity = min(current_state_id_b - action, self.maximum_services)

        assert a_new_capacity >= 0, "action is not valid "
        assert b_new_capacity >= 0, "action is not valid "

        expected_reward += np.abs(action) * self.rent_punishment

        # profile = cProfile.Profile()
        # profile.enable()
        for demand_on_a in range(0, self.poisson_upper_bound):
            for demand_on_b in range(0, self.poisson_upper_bound):

                rent_probability = self.poisson_calc_a_demand.pmf(k=demand_on_a) * self.poisson_calc_b_demand.pmf(k=demand_on_b)

                reward = 0
                if demand_on_a > a_new_capacity:
                    total_rented_a = a_new_capacity
                    reward += (total_rented_a * self.service_reward) + (
                            (demand_on_a - total_rented_a) * self.no_service_punishment)
                else:
                    # demand_on_a <= a_new_capacity:
                    total_rented_a = demand_on_a
                    reward += (total_rented_a * self.service_reward)

                if demand_on_b > b_new_capacity:
                    total_rented_b = b_new_capacity
                    reward += (total_rented_b * self.service_reward) + (
                                (demand_on_b - total_rented_b) * self.no_service_punishment)
                else:
                    #  demand_on_b <= b_new_capacity
                    total_rented_b = demand_on_b
                    reward += (total_rented_b * self.service_reward)

                for returned_to_a in range(0, self.poisson_upper_bound):
                    for returned_to_b in range(0, self.poisson_upper_bound):
                        probability = self.poisson_calc_a_cancellation.pmf(k=returned_to_a) * self.poisson_calc_b_cancellation.pmf(k=returned_to_b) * rent_probability

                        next_state_cars_in_a = min(a_new_capacity + returned_to_a - total_rented_a, self.maximum_services)
                        next_state_cars_in_b = min(b_new_capacity + returned_to_b - total_rented_b, self.maximum_services)

                        expected_reward += probability * (reward + self.discount_factor * state_values[next_state_cars_in_a, next_state_cars_in_b])
        # profile.disable()
        # ps = pstats.Stats(profile)
        # ps.print_stats()

        return expected_reward
