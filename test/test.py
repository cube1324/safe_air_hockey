from mushroom_rl.environments.pybullet_envs.air_hockey import AirHockeyHit

from mushroom_rl.algorithms.actor_critic import DDPG, TD3
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.utils.dataset import compute_J

import torch.optim as optim
import torch.nn.functional as F


import numpy as np
import time

from nets import ActorNetwork, CriticNetwork

from tqdm import trange


logger = Logger("ddpg", results_dir=None)


# MDP
gamma = 0.99
env = AirHockeyHit(debug_gui=False, random_init=True)


# Policy
policy_class = OrnsteinUhlenbeckPolicy
policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

# Settings
initial_replay_size = 10**5
max_replay_size = 10**6
batch_size = 200
n_features = 80
tau = .001
use_cuda = False



 # Approximator
actor_input_shape = env.info.observation_space.shape
actor_params = dict(network=ActorNetwork,
                    n_features=n_features,
                    input_shape=actor_input_shape,
                    output_shape=env.info.action_space.shape,
                    use_cuda=use_cuda)

actor_optimizer = {'class': optim.Adam,
                    'params': {'lr': 10**-3}}

critic_input_shape = (actor_input_shape[0] + env.info.action_space.shape[0],)
critic_params = dict(network=CriticNetwork,
                        optimizer={'class': optim.Adam,
                                'params': {'lr': 10**-3}},
                        loss=F.mse_loss,
                        n_features=n_features,
                        input_shape=critic_input_shape,
                        output_shape=(1,),
                        use_cuda=use_cuda)

# Agent
agent = DDPG(env.info, policy_class, policy_params,
            actor_params, actor_optimizer, critic_params, batch_size,
            initial_replay_size, max_replay_size, tau)

# Algorithm
core = Core(agent, env)

core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

n_steps_test=20000
n_steps=10**5
n_epochs=40

# RUN
dataset = core.evaluate(n_steps=n_steps_test, render=False)
J = np.mean(compute_J(dataset, gamma))
R = np.mean(compute_J(dataset))

logger.epoch_info(0, J=J, R=R)

for n in trange(n_epochs, leave=False):
    core.learn(n_steps=n_steps, n_steps_per_fit=30**3)
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = np.mean(compute_J(dataset, gamma))
    R = np.mean(compute_J(dataset))

    logger.epoch_info(n+1, J=J, R=R)

core.evaluate(n_episodes=5, render=True)

agent.policy.save("test")