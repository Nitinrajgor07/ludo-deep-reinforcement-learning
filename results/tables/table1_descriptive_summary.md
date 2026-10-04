### Table 1: Master Descriptive Statistics across Stored Evaluations

| Algorithm   | Opponent   | Metric            |   N |     Mean |   StdDev |   Variance | Median (IQR)   | 95% CI               | CI Method                      |
|:------------|:-----------|:------------------|----:|---------:|---------:|-----------:|:---------------|:---------------------|:-------------------------------|
| DQN         | EXPERT     | win_rate          | 250 |   0.156  |   0.3629 |     0.1317 | 0.00 (0.00)    | [0.1163, 0.2061]     | Wilson Score Interval (95%)    |
| DQN         | EXPERT     | agent_reward      | 250 |   4.92   |   5.5143 |    30.4078 | 2.25 (10.79)   | [4.2331, 5.6069]     | Student's t-Distribution (95%) |
| DQN         | EXPERT     | turns             | 250 | 169.956  |  25.4619 |   648.307  | 167.00 (34.75) | [166.7844, 173.1276] | Student's t-Distribution (95%) |
| DQN         | EXPERT     | agent_captures    | 250 |   0.636  |   0.8732 |     0.7626 | 0.00 (1.00)    | [0.5272, 0.7448]     | Student's t-Distribution (95%) |
| DQN         | EXPERT     | agent_tokens_home | 250 |   1.968  |   1.2349 |     1.5251 | 2.00 (2.00)    | [1.8142, 2.1218]     | Student's t-Distribution (95%) |
| DQN         | RANDOM     | win_rate          | 250 |   0.492  |   0.4999 |     0.2499 | 0.00 (1.00)    | [0.4306, 0.5536]     | Wilson Score Interval (95%)    |
| DQN         | RANDOM     | agent_reward      | 250 |   5.106  |   6.057  |    36.6876 | 2.77 (11.69)   | [4.3515, 5.8605]     | Student's t-Distribution (95%) |
| DQN         | RANDOM     | turns             | 250 | 174.652  |  22.8108 |   520.332  | 173.00 (30.00) | [171.8106, 177.4934] | Student's t-Distribution (95%) |
| DQN         | RANDOM     | agent_captures    | 250 |   1.348  |   1.2136 |     1.4728 | 1.00 (2.00)    | [1.1968, 1.4992]     | Student's t-Distribution (95%) |
| DQN         | RANDOM     | agent_tokens_home | 250 |   3.04   |   1.1187 |     1.2514 | 3.00 (2.00)    | [2.9007, 3.1793]     | Student's t-Distribution (95%) |
| DQN         | SELF_PLAY  | win_rate          | 250 |   0.484  |   0.4997 |     0.2497 | 0.00 (1.00)    | [0.4228, 0.5457]     | Wilson Score Interval (95%)    |
| DQN         | SELF_PLAY  | agent_reward      | 250 |   5.1186 |   5.9222 |    35.0725 | 2.75 (11.38)   | [4.3809, 5.8563]     | Student's t-Distribution (95%) |
| DQN         | SELF_PLAY  | turns             | 250 | 177.912  |  23.5415 |   554.201  | 174.00 (30.75) | [174.9796, 180.8444] | Student's t-Distribution (95%) |
| DQN         | SELF_PLAY  | agent_captures    | 250 |   0.672  |   0.8241 |     0.6791 | 0.00 (1.00)    | [0.5693, 0.7747]     | Student's t-Distribution (95%) |
| DQN         | SELF_PLAY  | agent_tokens_home | 250 |   3.064  |   1.0958 |     1.2007 | 3.00 (2.00)    | [2.9275, 3.2005]     | Student's t-Distribution (95%) |
| PPO         | RANDOM     | win_rate          | 250 |   0.44   |   0.4964 |     0.2464 | 0.00 (1.00)    | [0.3798, 0.5020]     | Wilson Score Interval (95%)    |
| PPO         | RANDOM     | agent_reward      | 250 |   4.8968 |   5.7542 |    33.1105 | 2.48 (11.30)   | [4.1800, 5.6136]     | Student's t-Distribution (95%) |
| PPO         | RANDOM     | turns             | 250 | 175.76   |  24.7112 |   610.641  | 173.50 (35.75) | [172.6819, 178.8381] | Student's t-Distribution (95%) |
| PPO         | RANDOM     | agent_captures    | 250 |   1.3    |   1.1556 |     1.3353 | 1.00 (2.00)    | [1.1561, 1.4439]     | Student's t-Distribution (95%) |
| PPO         | RANDOM     | agent_tokens_home | 250 |   3.224  |   0.8298 |     0.6886 | 3.00 (1.00)    | [3.1206, 3.3274]     | Student's t-Distribution (95%) |

*Source: Experimental results stored in results/raw/ adhering to Final Experiment Protocol Section 20.*
