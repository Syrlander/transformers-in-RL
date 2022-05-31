from rl_thesis.models import BaseModelConfig


class Config(BaseModelConfig):
    # Strategy to update weights see https://arxiv.org/pdf/1507.06527.pdf section "Stable Recurrent Updates" _RANDOM_ORDER_UPDATE refers to the regular Q-learning update strategy with completely random samples
    RANDOM_ORDER_UPDATE, SEQUENTIAL_UPDATE_FROM_BEGINNING, SEQUENTIAL_UPDATE_RANDOM_START = 0, 1, 2
    DECAY_LINEAR, DECAY_EXP = 0, 1
    # After how many steps should target net be updated with policy net's parameters
    target_update = 1000

    eps_start = 0.9
    eps_end = 0.05
    # when using exponential decay this controls how fast eps converges towards eps_end (higher value means slower decay)
    eps_decay = 2000
    # When using linear decay this is the step from which the function will return eps_end forever after
    eps_end_step = 24000
    decay_function = 1
    update_sampling_strategy = RANDOM_ORDER_UPDATE
    batch_size = 2
    # only used if update_sampling_strategy is SEQUENTIAL_UPDATE_RANDOM_START
    n_time_steps_per_sample = 25
    final_state_value = -1
    optimizer = "RMSprop"
    learning_rate = 1e-2
    gamma = 0.999
    learning_starts = 1000
    # After how many steps should the model be optimized
    optimize_freq = 200
    gradient_steps = 1

    # Size of replay experience buffer
    replay_total_timesteps = 10000
    # How often to use prioritized replay in buffer
    prioritized_replay_factor = 0.2
    # How much reward before episode is considered priority
    priority_min_reward = 0

    flatten_state = False
    # how often to save model without evaluating first
    checkpoint_freq = 10000

    def __init__(self):
        super().__init__()
        [
            setattr(self, k, v) for k, v in vars(Config).items()
            if not k.startswith("_")
        ]