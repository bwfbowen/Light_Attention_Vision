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
os.environ.setdefault('JAX_PLATFORM_NAME', 'GPU')     # tell JAX to use GPU/TPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet

config = {
    'name': 'ppo_conv',
    'TMAX': 3000000,
    'ER_beta': 0.001,
    'lr': 3e-4,
    'Nstep_n': 5,
    'Nstep_gamma': 0.99,
    'RB_capacity': 256,
    'save_T_init': 50000,
    'save_T_period': 50000,
    'env_name': 'PongNoFrameskip-v4',
    'num_frames': 3,
    'max_episode_steps': 108000 // 3,
    'tensorboard_dir': 'drive/MyDrive/data/tensorboard/'
}

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
if config['name'] == 'ppo_conv':
  def shared(S, is_training):
      seq = hk.Sequential([
          coax.utils.diff_transform,
          hk.Conv2D(32, kernel_shape=8, stride=4, padding='VALID'), jax.nn.relu,
          hk.Conv2D(64, kernel_shape=4, stride=2, padding='VALID'), jax.nn.relu,
          hk.Conv2D(64, kernel_shape=3, stride=1, padding='VALID'), jax.nn.relu,
          hk.Flatten(),
      ])
      X = jnp.stack(S, axis=-1) / 255.  # stack frames
      # tmp = seq(X)
      # print(tmp.shape)
      return seq(X)

elif config['name'] == 'ppo_regular_attn':
  def shared(S, is_training):
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
      ])
      X = jnp.stack(S, axis=-1) / 255.  # stack frames
      # tmp = seq(X)
      # print(tmp.shape)
      return seq(X)

elif config['name'] == 'ppo_fast_attn':
  def shared(S, is_training):
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
                     name='regular_attn'),
          hk.Flatten()           
      ])
      
      X = jnp.stack(S, axis=-1) / 255.  # stack frames
      # tmp = seq(X)
      # print(tmp.shape)
      return seq(X)

def func_pi(S, is_training):
    logits = hk.Sequential((
        hk.Linear(256), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros),
        jax.nn.softmax
    ))
    X = shared(S, is_training)
    return {'logits': logits(X)}


def func_v(S, is_training):
    value = hk.Sequential((
        hk.Linear(256), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    X = shared(S, is_training)
    return value(X)

def ppo_train(func_pi, func_v, config):
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
    pi = coax.Policy(func_pi, env)
    v = coax.V(func_v, env)

    # target networks
    pi_behavior = pi.copy()
    v_targ = v.copy()

    # policy regularizer (avoid premature exploitation)
    entropy = coax.regularizers.EntropyRegularizer(pi, beta=config['ER_beta'])

    # updaters
    simpletd = coax.td_learning.SimpleTD(v, v_targ, optimizer=adam(config['lr']))
    ppo_clip = coax.policy_objectives.PPOClip(pi, 
                                              regularizer=entropy, 
                                              optimizer=adam(config['lr']))

    # reward tracer and replay buffer
    tracer = coax.reward_tracing.NStep(n=config['Nstep_n'], gamma=config['Nstep_gamma'])
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=config['RB_capacity'])

    while env.T < config['TMAX']:
        # check = defaultdict(int)
        s, info = env.reset()

        for t in range(env.spec.max_episode_steps):
            a, logp = pi_behavior(s, return_logp=True)
            # check[a] += 1
            s_next, r, done, truncated, info = env.step(a)

            # trace rewards and add transition to replay buffer
            tracer.add(s, a, r, done, logp)
            while tracer:
                buffer.add(tracer.pop())

            # learn
            if len(buffer) >= buffer.capacity:
                num_batches = int(4 * buffer.capacity / 32)  # 4 epochs per round
                for _ in range(num_batches):
                    transition_batch = buffer.sample(32)
                    metrics_v, td_error = simpletd.update(transition_batch, return_td_error=True)
                    metrics_pi = ppo_clip.update(transition_batch, td_error)
                    env.record_metrics(metrics_v)
                    env.record_metrics(metrics_pi)

                buffer.clear()

                # sync target networks
                pi_behavior.soft_update(pi, tau=0.1)
                v_targ.soft_update(v, tau=0.1)

            if done or truncated:
                break

            s = s_next
        # print(check)
        # print(pi.params)
        if (
            env.period(name='generate_gif', T_period=config['save_T_period'])
            and 
            env.T > config['save_T_init']
            ):
            coax.utils.dump(pi_behavior, 'pi_behavior')

if __name__ == '__main__':
    # test
    s, info = env.reset()
    pi = coax.Policy(func_pi, env)
    v = coax.V(func_v, env)
    pi_behavior = pi.copy()
    pi_behavior(s, return_logp=True)