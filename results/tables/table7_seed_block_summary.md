### Table 7: Empirical Seed Block Invariance Breakdown

| Algorithm   | Opponent   | Seed Range   | Seed Block                   | Metric       |   N |   Mean |   Variance |
|:------------|:-----------|:-------------|:-----------------------------|:-------------|----:|-------:|-----------:|
| DQN         | EXPERT     | 1042..1091   | seed_block_1 (games 0-49)    | win_rate     |  50 |  0.22  |     0.1716 |
| DQN         | EXPERT     | 1042..1091   | seed_block_1 (games 0-49)    | agent_reward |  50 |  5.334 |    34.7392 |
| DQN         | EXPERT     | 1092..1141   | seed_block_2 (games 50-99)   | win_rate     |  50 |  0.08  |     0.0736 |
| DQN         | EXPERT     | 1092..1141   | seed_block_2 (games 50-99)   | agent_reward |  50 |  4.96  |    31.5777 |
| DQN         | EXPERT     | 1142..1191   | seed_block_3 (games 100-149) | win_rate     |  50 |  0.06  |     0.0564 |
| DQN         | EXPERT     | 1142..1191   | seed_block_3 (games 100-149) | agent_reward |  50 |  4.718 |    26.3688 |
| DQN         | EXPERT     | 1192..1241   | seed_block_4 (games 150-199) | win_rate     |  50 |  0.26  |     0.1924 |
| DQN         | EXPERT     | 1192..1241   | seed_block_4 (games 150-199) | agent_reward |  50 |  4.766 |    31.4483 |
| DQN         | EXPERT     | 1242..1291   | seed_block_5 (games 200-249) | win_rate     |  50 |  0.16  |     0.1344 |
| DQN         | EXPERT     | 1242..1291   | seed_block_5 (games 200-249) | agent_reward |  50 |  4.822 |    30.1349 |
| DQN         | RANDOM     | 1042..1091   | seed_block_1 (games 0-49)    | win_rate     |  50 |  0.5   |     0.25   |
| DQN         | RANDOM     | 1042..1091   | seed_block_1 (games 0-49)    | agent_reward |  50 |  5.012 |    37.4065 |
| DQN         | RANDOM     | 1092..1141   | seed_block_2 (games 50-99)   | win_rate     |  50 |  0.5   |     0.25   |
| DQN         | RANDOM     | 1092..1141   | seed_block_2 (games 50-99)   | agent_reward |  50 |  5.18  |    37.9298 |
| DQN         | RANDOM     | 1142..1191   | seed_block_3 (games 100-149) | win_rate     |  50 |  0.4   |     0.24   |
| DQN         | RANDOM     | 1142..1191   | seed_block_3 (games 100-149) | agent_reward |  50 |  5.222 |    37.8019 |
| DQN         | RANDOM     | 1192..1241   | seed_block_4 (games 150-199) | win_rate     |  50 |  0.5   |     0.25   |
| DQN         | RANDOM     | 1192..1241   | seed_block_4 (games 150-199) | agent_reward |  50 |  5.007 |    35.266  |
| DQN         | RANDOM     | 1242..1291   | seed_block_5 (games 200-249) | win_rate     |  50 |  0.56  |     0.2464 |
| DQN         | RANDOM     | 1242..1291   | seed_block_5 (games 200-249) | agent_reward |  50 |  5.109 |    37.9905 |
| DQN         | SELF_PLAY  | 1042..1091   | seed_block_1 (games 0-49)    | win_rate     |  50 |  0.4   |     0.24   |
| DQN         | SELF_PLAY  | 1042..1091   | seed_block_1 (games 0-49)    | agent_reward |  50 |  5.06  |    35.5647 |
| DQN         | SELF_PLAY  | 1092..1141   | seed_block_2 (games 50-99)   | win_rate     |  50 |  0.44  |     0.2464 |
| DQN         | SELF_PLAY  | 1092..1141   | seed_block_2 (games 50-99)   | agent_reward |  50 |  5.289 |    36.0752 |
| DQN         | SELF_PLAY  | 1142..1191   | seed_block_3 (games 100-149) | win_rate     |  50 |  0.5   |     0.25   |
| DQN         | SELF_PLAY  | 1142..1191   | seed_block_3 (games 100-149) | agent_reward |  50 |  5.205 |    35.6955 |
| DQN         | SELF_PLAY  | 1192..1241   | seed_block_4 (games 150-199) | win_rate     |  50 |  0.58  |     0.2436 |
| DQN         | SELF_PLAY  | 1192..1241   | seed_block_4 (games 150-199) | agent_reward |  50 |  4.904 |    34.4913 |
| DQN         | SELF_PLAY  | 1242..1291   | seed_block_5 (games 200-249) | win_rate     |  50 |  0.5   |     0.25   |
| DQN         | SELF_PLAY  | 1242..1291   | seed_block_5 (games 200-249) | agent_reward |  50 |  5.135 |    36.3106 |
| PPO         | RANDOM     | 1042..1091   | seed_block_1 (games 0-49)    | win_rate     |  50 |  0.5   |     0.25   |
| PPO         | RANDOM     | 1042..1091   | seed_block_1 (games 0-49)    | agent_reward |  50 |  4.603 |    32.8032 |
| PPO         | RANDOM     | 1092..1141   | seed_block_2 (games 50-99)   | win_rate     |  50 |  0.28  |     0.2016 |
| PPO         | RANDOM     | 1092..1141   | seed_block_2 (games 50-99)   | agent_reward |  50 |  5.065 |    32.8156 |
| PPO         | RANDOM     | 1142..1191   | seed_block_3 (games 100-149) | win_rate     |  50 |  0.5   |     0.25   |
| PPO         | RANDOM     | 1142..1191   | seed_block_3 (games 100-149) | agent_reward |  50 |  4.826 |    33.2122 |
| PPO         | RANDOM     | 1192..1241   | seed_block_4 (games 150-199) | win_rate     |  50 |  0.4   |     0.24   |
| PPO         | RANDOM     | 1192..1241   | seed_block_4 (games 150-199) | agent_reward |  50 |  4.796 |    32.7323 |
| PPO         | RANDOM     | 1242..1291   | seed_block_5 (games 200-249) | win_rate     |  50 |  0.52  |     0.2496 |
| PPO         | RANDOM     | 1242..1291   | seed_block_5 (games 200-249) | agent_reward |  50 |  5.194 |    36.4697 |

*Source: Experimental results stored in results/raw/ adhering to Final Experiment Protocol Section 20.*
