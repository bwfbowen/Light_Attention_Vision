import os

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from optax import adam

from google.colab import drive
from light_vision_attention import VisionAttn

drive.mount('/content/drive')

# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


config = {
    'name': 'dqn_regular_attn',
    'TMAX': 3000000,
    'temperature': 0.015,
    'ER_beta': 0.001,
    'lr': 3e-4,
    'Nstep_n': 5,
    'Nstep_gamma': 0.99,
    'RB_capacity': 1000000,
    'RB_alpha': 0.6,
    'RB_warmup': 50000,
    'batch_size': 32,
    'save_T_init': 50000,
    'save_T_period': 50000,
    'soft_update_tau': 1,
    'update_freq': 10000,
    'learn_freq': 4,
    'env_name': 'PongNoFrameskip-v4',
    'num_frames': 3,
    'max_episode_steps': 108000 // 3,
    'tensorboard_dir': 'drive/MyDrive/data/tensorboard/'
}


# env with preprocessing
def make_env(config):
  name = config['name']
  # env with preprocessing
  env = gym.make(config['env_name'], render_mode='rgb_array')
  env = gym.wrappers.AtariPreprocessing(env)
  env = coax.wrappers.FrameStacking(env, num_frames=config['num_frames'])
  env = gym.wrappers.TimeLimit(env, max_episode_steps=config['max_episode_steps'])
  env = coax.wrappers.TrainMonitor(env, 
                                    name=name, 
                                    tensorboard_dir=os.path.join(config['tensorboard_dir'],
                                                                name))
  return env

env = make_env(config)

if config['name'] == 'dqn_conv':
  def func(S, is_training):
      """ type-2 q-function: s -> q(s,.) """
      seq = hk.Sequential((
          coax.utils.diff_transform,
          hk.Conv2D(32, kernel_shape=8, stride=4, padding='VALID'), jax.nn.relu,
          hk.Conv2D(64, kernel_shape=4, stride=2, padding='VALID'), jax.nn.relu,
          hk.Conv2D(64, kernel_shape=3, stride=1, padding='VALID'), jax.nn.relu,
          hk.Flatten(),
          hk.Linear(256), jax.nn.relu,
          hk.Linear(env.action_space.n, w_init=jnp.zeros),
      ))
      X = jnp.stack(S, axis=-1) / 255.  # stack frames
      return seq(X)

elif config['name'] == 'dqn_fast_attn':
  def func(S, is_training):
      """ type-2 q-function: s -> q(s,.) """
      seq = hk.Sequential([
          coax.utils.diff_transform,
          hk.Conv2D(32, kernel_shape=8, stride=4, padding='VALID'), jax.nn.relu,
          hk.Conv2D(64, kernel_shape=4, stride=2, padding='VALID'), jax.nn.relu,
          hk.Conv2D(64, kernel_shape=3, stride=1, padding='VALID'), jax.nn.relu,
          VisionAttn(embed_dim=64,
                     hidden_dim=128,
                     num_heads=8,
                     num_layers=2,
                     num_patches=64,
                     dropout_prob=0.1,
                     use_fask_attn=True,
                     name='fast_attn'),
          hk.Flatten(),
          hk.Linear(256), jax.nn.relu,
          hk.Linear(env.action_space.n, w_init=jnp.zeros),          
      ])
      X = jnp.stack(S, axis=-1) / 255.  # stack frames
      return seq(X)

elif config['name'] == 'dqn_regular_attn':
  def func(S, is_training):
      """ type-2 q-function: s -> q(s,.) """
      seq = hk.Sequential([
          coax.utils.diff_transform,
          hk.Conv2D(32, kernel_shape=8, stride=4, padding='VALID'), jax.nn.relu,
          hk.Conv2D(64, kernel_shape=4, stride=2, padding='VALID'), jax.nn.relu,
          hk.Conv2D(64, kernel_shape=3, stride=1, padding='VALID'), jax.nn.relu,
          VisionAttn(embed_dim=64,
                     hidden_dim=128,
                     num_heads=8,
                     num_layers=2,
                     num_patches=64,
                     dropout_prob=0.1,
                     use_fask_attn=False,
                     name='regular_attn'),
          hk.Flatten(),
          hk.Linear(256), jax.nn.relu,
          hk.Linear(env.action_space.n, w_init=jnp.zeros),          
      ])
      X = jnp.stack(S, axis=-1) / 255.  # stack frames
      return seq(X)

def dqn_train(func, config):
    ########################
    # Environment
    ########################
    # the name of this training script
    name = config['name']

    # env with preprocessing
    env = make_env(config)
    ########################
    # Agent
    ########################
    # function approximators
    # function approximator
    q = coax.Q(func, env)
    pi = coax.BoltzmannPolicy(q, temperature=config['temperature'])  # <--- different from standard DQN (ε-greedy)

    # target network
    q_targ = q.copy()

    # updater
    qlearning = coax.td_learning.QLearning(q, q_targ=q_targ, optimizer=adam(config['lr']))

    # reward tracer and replay buffer
    tracer = coax.reward_tracing.NStep(n=config['Nstep_n'], gamma=config['Nstep_gamma'])
    buffer = coax.experience_replay.PrioritizedReplayBuffer(capacity=config['RB_capacity'], alpha=config['RB_alpha'])

    # schedule for the PER beta hyperparameter
    beta = coax.utils.StepwiseLinearFunction((0, 0.4), (1000000, 1))

    while env.T < config['TMAX']:
        s, info = env.reset()
        buffer.beta = beta(env.T)

        for t in range(env.spec.max_episode_steps):
            a = pi(s)
            s_next, r, done, truncated, info = env.step(a)

            # trace rewards and add transition to replay buffer
            tracer.add(s, a, r, done or truncated)
            while tracer:
                transition = tracer.pop()
                buffer.add(transition, qlearning.td_error(transition))

            # learn
            if env.T % config['learn_freq'] == 0 and len(buffer) > config['RB_warmup']:  # buffer warm-up
                transition_batch = buffer.sample(batch_size=config['batch_size'])
                metrics, td_error = qlearning.update(transition_batch, return_td_error=True)
                buffer.update(transition_batch.idx, td_error)
                env.record_metrics(metrics)

            if env.T % config['update_freq'] == 0:
                q_targ.soft_update(q, tau=config['soft_update_tau'])

            if done or truncated:
                break

            s = s_next

        # generate an animated GIF to see what's going on
        if env.period(name='generate_gif', T_period=config['save_T_period']) and env.T > config['save_T_init']:
            coax.utils.dump(pi, 'pi')
        
if __name__ == '__main__':
    # test
    s, info = env.reset()
    q = coax.Q(func, env)
    pi = coax.BoltzmannPolicy(q, temperature=config['temperature'])
    pi(s)