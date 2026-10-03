"""Model 3: Discrete Soft Actor-Critic (SAC) package."""

from phd_ludo_drl.agents.model3.sac_network import DiscreteActor, DiscreteCritic
from phd_ludo_drl.agents.model3.replay_buffer import DiscreteSACReplayBuffer
from phd_ludo_drl.agents.model3.sac_agent import DiscreteSACAgent
from phd_ludo_drl.agents.model3.trainer import DiscreteSACTrainer, DiscreteSACTrainingMetrics

__all__ = [
    "DiscreteActor",
    "DiscreteCritic",
    "DiscreteSACReplayBuffer",
    "DiscreteSACAgent",
    "DiscreteSACTrainer",
    "DiscreteSACTrainingMetrics",
]
