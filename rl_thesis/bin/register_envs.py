from gym.envs import register


def register_envs():
    register(
        "numpad_discrete-v1",
        entry_point="rl_thesis.environments.numpad_discrete:Environment",
    )
    register(
        "numpad_continuous-v1",
        entry_point="rl_thesis.environments.numpad_continuous:Environment",
    )
    register(
        "numpad_continuous_discrete_actions-v1",
        entry_point=
        "rl_thesis.environments.numpad_continuous_discrete_actions:Environment",
    )
    register(
        "mountain_car_no_velocity-v1",
        entry_point=
        "rl_thesis.environments.mountain_car_no_velocity:Environment",
    )
    register(
        "mountain_car_screen_input-v1",
        entry_point=
        "rl_thesis.environments.mountain_car_screen_input:Environment",
    )
