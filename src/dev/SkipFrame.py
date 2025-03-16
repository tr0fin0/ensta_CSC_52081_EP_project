import gymnasium as gym

class SkipFrame(gym.Wrapper):
    """
    Gym environments custom wrapper to skip a specified number of frames.

    Attributes:
        env (gym.Env): The environment to wrap.
        _skip (int): The number of frames to skip.

    Methods:
        step(action):
            Repeats the given action for the specified number of frames and
            accumulates the reward.
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info
