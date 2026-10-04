### Table 6: Distributional Characteristics & Normality Testing

| Algorithm   | Opponent   | Metric            |   N |   Skewness |   Kurtosis |   Shapiro-Wilk W |   Shapiro-Wilk p | Normal (alpha=0.05)   | Method Justification                            |
|:------------|:-----------|:------------------|----:|-----------:|-----------:|-----------------:|-----------------:|:----------------------|:------------------------------------------------|
| DQN         | EXPERT     | agent_reward      | 250 |     0.2117 |    -1.7052 |           0.8102 |       9.1585e-17 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | EXPERT     | turns             | 250 |     0.5371 |     0.1169 |           0.9789 |       0.0009     | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | EXPERT     | agent_captures    | 250 |     1.3924 |     1.7093 |           0.7266 |       5.3132e-20 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | EXPERT     | agent_tokens_home | 250 |     0.2275 |    -0.9694 |           0.9014 |       9.7212e-12 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | RANDOM     | agent_reward      | 250 |     0.1398 |    -1.7824 |           0.8043 |       5.0203e-17 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | RANDOM     | turns             | 250 |     0.612  |     1.3187 |           0.9785 |       0.0008     | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | RANDOM     | agent_captures    | 250 |     1.1609 |     2.0807 |           0.8596 |       2.5093e-14 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | RANDOM     | agent_tokens_home | 250 |    -0.8204 |    -0.4492 |           0.7949 |       1.9812e-17 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | SELF_PLAY  | agent_reward      | 250 |     0.1211 |    -1.8116 |           0.7943 |       1.87e-17   | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | SELF_PLAY  | turns             | 250 |     0.769  |     0.6878 |           0.9641 |       6.5301e-06 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | SELF_PLAY  | agent_captures    | 250 |     1.1017 |     0.7503 |           0.7624 |       1.0103e-18 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| DQN         | SELF_PLAY  | agent_tokens_home | 250 |    -0.9343 |    -0.0162 |           0.7958 |       2.1735e-17 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| PPO         | RANDOM     | agent_reward      | 250 |     0.0988 |    -1.8407 |           0.7894 |       1.1753e-17 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| PPO         | RANDOM     | turns             | 250 |     0.474  |    -0.1439 |           0.981  |       0.002      | No (Non-normal)       | Parametric & non-parametric tests both reported |
| PPO         | RANDOM     | agent_captures    | 250 |     0.6963 |    -0.1032 |           0.8764 |       2.3367e-13 | No (Non-normal)       | Parametric & non-parametric tests both reported |
| PPO         | RANDOM     | agent_tokens_home | 250 |    -0.8594 |     0.0708 |           0.7968 |       2.3886e-17 | No (Non-normal)       | Parametric & non-parametric tests both reported |

*Source: Experimental results stored in results/raw/ adhering to Final Experiment Protocol Section 20.*
