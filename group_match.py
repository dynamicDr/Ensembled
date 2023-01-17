import os
import re

import match


def extract_step_numbers(exp_number):
    step_numbers = []
    for file_name in os.listdir(f"./models/SSLShootEnv/{exp_number}"):
        match = re.search(r'(?<=_)\d+(?=k_actor)', file_name)
        if match:
            step_numbers.append(int(match.group()))
    return sorted(step_numbers)

if __name__ == '__main__':
    number = 39
    max_episode = 1000
    display = False

    exp_numbers = extract_step_numbers(f"./{number}")
    print(exp_numbers)
    for exp in exp_numbers:
        goal_num, opp_num, done_ball_out, done_rbt_out, done_other, avg_episode_step = match.match(number, exp,
                                                                                                   max_episode,
                                                                                                   display)
        print("\nexp_x_k_step",exp,
              "\ngoal", goal_num,
              "\nopp_goal", opp_num,
              "\ndone_ball_out", done_ball_out,
              "\ndone_rbt_out", done_rbt_out,
              "\ndone_other", done_other,
              "\navg_episode_step", avg_episode_step,
              "\n================================")

        with open(f'./evaluate/output_{number}.txt', 'a+', encoding='utf-8', errors='ignore') as f:
            text = f"\nexp_x_k_step {max_episode}\ngoal {goal_num}\nopp_goal {opp_num}\ndone_ball_out {done_ball_out}\ndone_rbt_out {done_rbt_out}\ndone_other {done_other}\navg_episode_step {avg_episode_step}\n================================"
            f.write(text)