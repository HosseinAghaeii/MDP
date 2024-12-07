import pygame

from environment import PygameInit, AngryBirds
from helper import Helper

if __name__ == "__main__":

    FPS = 10
    env = AngryBirds()
    screen, clock = PygameInit.initialization()
    state = env.reset()

    h = Helper(8, env.transition_table)
    allGoals = env.select_goals_and_pigs()
    policy = {}

    for goal in allGoals:
        tr_table = env.generate_transition_tables_for_goal(goal)
        Helper.tr_table_setter(h, tr_table)
        v, q = Helper.value_iteration(h)
        new_policy = Helper.derive_policy(h, q)
        Helper.visualize_value_function(new_policy, v)
        policy[goal] = new_policy

    i = 1
    for _ in range(5):
        print("########################## Round " + str(i) + "##########################")
        i = i + 1
        running = True
        all_rewards = 0
        goal_index = 0
        curr_goal = allGoals[goal_index]
        curr_policy = policy[curr_goal]
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(screen)

            # extract action from policy
            action = curr_policy[state]
            next_state, probability, reward_episode, done = env.step(action)
            all_rewards += reward_episode
            if done:
                print(f"Episode finished with reward: {all_rewards}")
                state = env.reset()
                running = False
            if state == curr_goal and curr_goal != (7, 7):
                print(curr_goal, " now is reached. yummmmmmmm!")
                goal_index += 1
                curr_goal = allGoals[goal_index]
                curr_policy = policy[curr_goal]

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()
