### Table 5: Ablation Impact Across Representation and Reward Shaping

| Study Dimension         | Ablated Component / Variant   | Opponent   |   Games (N) | Win Rate (%)   |   Average Reward |   Average Turns |
|:------------------------|:------------------------------|:-----------|------------:|:---------------|-----------------:|----------------:|
| Representation Ablation | Flat 60-D Baseline            | Expert     |          50 | 18.0%          |             5.64 |           169.4 |
| Representation Ablation | Flat 60-D Baseline            | Random     |          50 | 52.0%          |             5.39 |           168.2 |
| Reward Shaping Ablation | Sparse Terminal (+/-1.0)      | Expert     |          50 | 20.0%          |             0.04 |           172.9 |
| Reward Shaping Ablation | Sparse Terminal (+/-1.0)      | Random     |          50 | 52.0%          |             0.08 |           174.2 |
| Control Baseline        | Structured CNN (Dense Reward) | Expert     |          50 | 14.0%          |             4.98 |           167.8 |
| Control Baseline        | Structured CNN (Dense Reward) | Random     |          50 | 42.0%          |             4.81 |           171.5 |

*Generated automatically from verified experimental data adhering to Final Experiment Protocol.*
