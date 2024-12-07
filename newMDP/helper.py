import numpy as np
import matplotlib.pyplot as matplot


class Helper:
    def __init__(self, num_states, tr_table):
        self.tr_table = tr_table
        self.num_states = num_states

    def tr_table_setter(self, tr_table):
        self.tr_table = tr_table

    def value_iteration(self):
        gamma = 0.9
        threshold = 0.0
        max_iteration = 2000

        v = np.zeros((8, 8))
        q = np.zeros((len(self.tr_table), 4))

        for iteration in range(max_iteration):
            max_difference = 0
            state_index = 0

            for current_state in self.tr_table.keys():
                old_value = v[current_state]

                for current_action in self.tr_table[current_state].keys():
                    q_value = 0
                    highest_probability = -1
                    best_next_state = current_state

                    for prob, next_state, reward in self.tr_table[current_state][current_action]:
                        if highest_probability < prob:
                            highest_probability = prob
                            best_next_state = next_state
                        q_value += prob * (reward + gamma * (v[next_state]))

                    if best_next_state == current_state:
                        q_value = -1e7  # Penalize staying in the same state

                    q[state_index][current_action] = q_value

                v[current_state] = max(q[state_index])
                max_difference = max(max_difference, abs(old_value - v[current_state]))
                state_index += 1

            if max_difference <= threshold:
                print('Algorithm has converged successfully.')
                return v, q

        print('Maximum number of iterations reached without convergence.')

        return v, q

    def derive_policy(self, q):
        total_states = 0
        optimal_policy = np.zeros((8, 8), dtype=int)

        for index, current_state in enumerate(self.tr_table.keys()):
            best_action = np.argmax(q[index])
            optimal_policy[current_state] = best_action
            total_states += 1

        return optimal_policy

    # heat map
    def visualize_value_function(self, optimal_values):
        fig, ax = matplot.subplots()
        heatmap = ax.imshow(optimal_values, cmap='plasma', origin='upper')
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_label('Value Score')
        ax.set_title('Heatmap of Optimal Values')
        matplot.show()

    def display_policy_and_values(self, optimal_policy, optimal_values):

        directions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        grid_x, grid_y = np.meshgrid(np.arange(8), np.arange(8))
        arrow_x, arrow_y = np.zeros_like(grid_x, dtype=float), np.zeros_like(grid_y, dtype=float)

        for state_id in self.tr_table.keys():
            selected_action = optimal_policy[state_id]
            arrow_x[state_id], arrow_y[state_id] = directions[selected_action]

        matplot.imshow(optimal_values, cmap='plasma', origin='upper')
        matplot.colorbar(label='Value Score')

        matplot.quiver(grid_y, grid_x, arrow_y, -arrow_x, angles='xy', scale_units='xy', scale=1, color='black')
        matplot.title('Optimal Policy with Value Function')
        matplot.show()

