### Table 2: Training Convergence and Computational Resource Profiles

| Algorithm    |   Training Steps |   Episodes | Training Win Rate   |   Average Reward |   Initial Loss |   Final Loss | Loss Reduction   |   Runtime (s) |   Peak Memory (MB) |
|:-------------|-----------------:|-----------:|:--------------------|-----------------:|---------------:|-------------:|:-----------------|--------------:|-------------------:|
| DQN          |              500 |          6 | 50.0%               |            11.82 |         0.027  |       0.0088 | 67.4%            |          8.73 |              316.4 |
| PPO          |              500 |          6 | 50.0%               |            11.98 |         0.8762 |       0.504  | 42.5%            |          1.36 |              321.2 |
| Discrete SAC |              500 |          6 | 50.0%               |            11.18 |         0.0666 |       0.0213 | 68.0%            |         14.38 |              326.6 |
| C51          |              500 |          6 | 33.3%               |            11.74 |         3.9335 |       3.9269 | 0.2%             |          5.53 |              331.3 |

*Generated automatically from verified experimental data adhering to Final Experiment Protocol.*
