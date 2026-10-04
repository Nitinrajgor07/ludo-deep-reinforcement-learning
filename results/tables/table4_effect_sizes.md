### Table 4: Measured Effect Sizes and Standardized Magnitudes

| Comparison Category   | Group 1      | Group 2         | Metric            | Effect Measure                  |   Point Estimate |   95% CI Lower |   95% CI Upper | Interpretation                                        |
|:----------------------|:-------------|:----------------|:------------------|:--------------------------------|-----------------:|---------------:|---------------:|:------------------------------------------------------|
| algorithm_contrast    | DQN (random) | PPO (random)    | win_rate          | Risk Difference (Win Rate Diff) |           0.052  |        -0.0353 |         0.1393 | contains 0 (not statistically distinguishable from 0) |
| algorithm_contrast    | DQN (random) | PPO (random)    | win_rate          | Odds Ratio                      |           1.2326 |         0.867  |         1.7525 | contains 1 (not distinguishable from 1.0)             |
| algorithm_contrast    | DQN (random) | PPO (random)    | agent_reward      | Cohen's d                       |           0.0354 |        -0.1399 |         0.2107 | negligible                                            |
| algorithm_contrast    | DQN (random) | PPO (random)    | agent_reward      | Hedges' g                       |           0.0354 |        -0.1397 |         0.2104 | negligible                                            |
| algorithm_contrast    | DQN (random) | PPO (random)    | turns             | Cohen's d                       |          -0.0466 |        -0.2219 |         0.1287 | negligible                                            |
| algorithm_contrast    | DQN (random) | PPO (random)    | turns             | Hedges' g                       |          -0.0465 |        -0.2216 |         0.1285 | negligible                                            |
| algorithm_contrast    | DQN (random) | PPO (random)    | agent_captures    | Cohen's d                       |           0.0405 |        -0.1348 |         0.2158 | negligible                                            |
| algorithm_contrast    | DQN (random) | PPO (random)    | agent_captures    | Hedges' g                       |           0.0404 |        -0.1346 |         0.2155 | negligible                                            |
| algorithm_contrast    | DQN (random) | PPO (random)    | agent_tokens_home | Cohen's d                       |          -0.1868 |        -0.3625 |        -0.0111 | negligible                                            |
| algorithm_contrast    | DQN (random) | PPO (random)    | agent_tokens_home | Hedges' g                       |          -0.1865 |        -0.362  |        -0.0111 | negligible                                            |
| opponent_contrast     | DQN (random) | DQN (expert)    | agent_reward      | Cohen's d                       |           0.0321 |        -0.1432 |         0.2074 | negligible                                            |
| opponent_contrast     | DQN (random) | DQN (expert)    | turns             | Cohen's d                       |           0.1943 |         0.0186 |         0.37   | negligible                                            |
| opponent_contrast     | DQN (random) | DQN (self_play) | agent_reward      | Cohen's d                       |          -0.0021 |        -0.1774 |         0.1732 | negligible                                            |
| opponent_contrast     | DQN (random) | DQN (self_play) | turns             | Cohen's d                       |          -0.1406 |        -0.3162 |         0.0349 | negligible                                            |
| opponent_contrast     | DQN (expert) | DQN (self_play) | agent_reward      | Cohen's d                       |          -0.0347 |        -0.21   |         0.1406 | negligible                                            |
| opponent_contrast     | DQN (expert) | DQN (self_play) | turns             | Cohen's d                       |          -0.3245 |        -0.5009 |        -0.148  | small                                                 |

*Source: Experimental results stored in results/raw/ adhering to Final Experiment Protocol Section 20.*
