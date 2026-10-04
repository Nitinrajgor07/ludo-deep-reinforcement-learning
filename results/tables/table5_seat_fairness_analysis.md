### Table 5: First-Mover / Seat Permutation Fairness Evaluation

| Algorithm   | Opponent   | Metric       |   Seat 0 Value |   Seat 1 Value |   Seat Bias (P0 - P1) |   Test Statistic |     p-Value | Fair at alpha=0.05   |
|:------------|:-----------|:-------------|---------------:|---------------:|----------------------:|-----------------:|------------:|:---------------------|
| DQN         | EXPERT     | win_rate     |         0.16   |          0.152 |                0.008  |           1.0627 | 1           | Yes (Neutral)        |
| DQN         | EXPERT     | agent_reward |        10.184  |         -0.344 |               10.528  |          51.6456 | 2.1348e-90  | No (Biased)          |
| DQN         | RANDOM     | win_rate     |         0.536  |          0.448 |                0.088  |           1.4233 | 0.2058      | Yes (Neutral)        |
| DQN         | RANDOM     | agent_reward |        10.958  |         -0.746 |               11.704  |          60.8335 | 9.3117e-107 | No (Biased)          |
| DQN         | SELF_PLAY  | win_rate     |         0.528  |          0.44  |                0.088  |           1.4237 | 0.2056      | Yes (Neutral)        |
| DQN         | SELF_PLAY  | agent_reward |        10.8652 |         -0.628 |               11.4932 |          65.5028 | 5.3065e-112 | No (Biased)          |
| PPO         | RANDOM     | win_rate     |         0.488  |          0.392 |                0.096  |           1.4783 | 0.1609      | Yes (Neutral)        |
| PPO         | RANDOM     | agent_reward |        10.4916 |         -0.698 |               11.1896 |          68.0509 | 3.6629e-119 | No (Biased)          |

*Source: Experimental results stored in results/raw/ adhering to Final Experiment Protocol Section 20.*
