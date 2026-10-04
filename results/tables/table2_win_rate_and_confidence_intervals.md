### Table 2: Win Rates and Rigorous Confidence Intervals (Wilson vs Bootstrap 10k)

| Algorithm   | Opponent   |   Games (N) | Win Rate (%)   |   Bernoulli Variance | SEM (%)   | 95% Wilson Score CI   | 95% Bootstrap CI   |
|:------------|:-----------|------------:|:---------------|---------------------:|:----------|:----------------------|:-------------------|
| DQN         | EXPERT     |         250 | 15.60%         |               0.1317 | 2.29%     | [11.63%, 20.61%]      | [11.20%, 20.40%]   |
| DQN         | RANDOM     |         250 | 49.20%         |               0.2499 | 3.16%     | [43.06%, 55.36%]      | [42.80%, 55.60%]   |
| DQN         | SELF_PLAY  |         250 | 48.40%         |               0.2497 | 3.16%     | [42.28%, 54.57%]      | [42.40%, 54.80%]   |
| PPO         | RANDOM     |         250 | 44.00%         |               0.2464 | 3.14%     | [37.98%, 50.20%]      | [37.99%, 50.00%]   |

*Source: Experimental results stored in results/raw/ adhering to Final Experiment Protocol Section 20.*
