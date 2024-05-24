from onpolicy.envs.HitBalls.HitBalls import PlankAndBall
import gym

class PlankAndBallENV(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world):

        self.world = world
        # # set required vectorized gym env property
        # self.n = len(world.policy_agents)
        self.action_space = self.world.action_space
        self.observation_space = self.world.observation_space
        self.share_observation_space = self.world.observation_space
        self.episode_length = self.world.episode_length

    def step(self, action_n):
        obs_n, reward_n, done_n, info_n = self.world.step(action_n)
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = self.world.reset()
        return obs_n