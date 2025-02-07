# -*- coding: utf-8 -*-
#Source Code of SQOG, 20240922

from typing import NamedTuple, Tuple
from dataclasses import dataclass
import random
import time
import d4rl
import gym
import numpy as np
import pyrallis  
from tqdm import tqdm
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from tensorflow_probability.substrates import jax as tfp
from tensorboardX import SummaryWriter
tfd = tfp.distributions
tfb = tfp.bijectors
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

@dataclass
class TrainArgs:
    # Experiment
    exp_name: str = "SQOG"
    env_id: str = "hopper"
    gym_id: str = env_id + '-' +'medium-v2'
    seed: int = 3
    log_dir: str = "runs_" + env_id
    total_iterations: int = int(1e6)
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256 
    expectile: float = 0.7
    temperature: float = 3.0
    polyak: float = 0.005
    eval_freq: int = int(5e3)
    eval_episodes: int = 10
    log_freq: int = 1000
    beta: float = 0.5
    alpha: float = 150

    normalized_q: bool = False
    normalize_states: bool = False
    
    def __post_init__(self):
        self.exp_name = f"{self.exp_name}__{self.gym_id}__alpha:{self.alpha}__beta:{self.beta}"


def make_env(env_id, seed):  
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def layer_init(scale=jnp.sqrt(2)):  
    return nn.initializers.orthogonal(scale)


class CriticNetwork(nn.Module):  
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        #trick
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=layer_init())(x)
        return x


class DoubleCriticNetwork(nn.Module):  
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        critic1 = CriticNetwork()(x, a)
        critic2 = CriticNetwork()(x, a)
        return critic1, critic2


# changed
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module): 
    action_dim: int
    max_action: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, temperature: float = 1.0):
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        action = nn.Dense(self.action_dim, kernel_init=layer_init())(x)
        action = nn.tanh(action) 
        return max_action * action


class TargetTrainState(TrainState):
    target_params: flax.core.FrozenDict


class Batch(NamedTuple):  
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array):
    return (states - mean) / std


class Dataset:  
    def __init__(self):
        self.size = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.masks = None
        self.next_observations = None

    def load(self, env, eps=1e-5,
            normalize: bool = False):
        self.env = env
        dataset = d4rl.qlearning_dataset(env)
        lim = 1 - eps  
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
        self.size = len(dataset["observations"])
        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32)[:, None]
        self.masks = 1.0 - dataset["terminals"].astype(np.float32)[:, None]
        self.next_observations = dataset["next_observations"].astype(
            np.float32)
    
        if normalize:
            self.mean, self.std = compute_mean_std(dataset["states"], eps=1e-3)
            dataset["states"] = normalize_states(
                dataset["states"], self.mean, self.std
            )
            dataset["next_states"] = normalize_states(
                dataset["next_states"], self.mean, self.std
            )
        '''if normalize_reward:
            dataset["rewards"] = Dataset.normalize_reward(dataset_name, buffer["rewards"])
        self.data = buffer'''


    def sample(self, batch_size):
        idx = np.random.randint(self.size, size=batch_size)
        data = (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.masks[idx],
            self.next_observations[idx],
        )
        return Batch(*data)


if __name__ == "__main__":
    # Logging setup
    args = pyrallis.parse(config_class=TrainArgs)
    print(vars(args))
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    # Eval env setup
    env = make_env(args.gym_id, args.seed)()
    assert isinstance(
        env.action_space, gym.spaces.Box), "only continuous action space is supported"
    observation = env.observation_space.sample()[np.newaxis]
    action = env.action_space.sample()[np.newaxis]
    max_action = float(env.action_space.high[0])

    # Agent setup
    actor = Actor(action_dim=np.prod(env.action_space.shape), max_action=max_action)
    actor_state = TargetTrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, observation),
        target_params=actor.init(actor_key, observation),
        tx=optax.adam(learning_rate=args.actor_lr)
    )
    qf = DoubleCriticNetwork()
    qf_state = TargetTrainState.create(
        apply_fn=qf.apply,
        params=qf.init(critic_key, observation, action),
        target_params=qf.init(critic_key, observation, action),
        tx=optax.adam(learning_rate=args.critic_lr)
    )

    # Dataset setup
    dataset = Dataset()
    dataset.load(env, args.normalize_states)
    start_time = time.time()

    # select an action using the distribution 
    @jax.jit
    def get_action(rng, actor_state, observation, temperature=1.0):
        action = actor.apply(actor_state.params, observation, temperature)
        rng, key = jax.random.split(rng)
        return rng, jnp.clip(action, -1, 1)


    def update_actor(actor_state, qf_state, batch, key):

        def actor_loss_fn(params):
            action = actor.apply(params, batch.observations)
            bc_penalty = ((action - batch.actions) ** 2).sum(-1)
            q1, q2 = qf.apply(qf_state.params,
                              batch.observations, action)
            q = jnp.minimum(q1, q2)
            alpha = args.alpha 
            lmbda = jax.lax.stop_gradient(alpha / jax.numpy.abs(q).mean())
            actor_loss = (bc_penalty - lmbda * q).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
            }

        (actor_loss, info), grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, info


    def update_qf(actor_state, qf_state, batch, key):
        next_action = actor.apply(actor_state.target_params, batch.next_observations, temperature=1.0)
        policy_noise_1 = 0.6 
        policy_noise_2 = 0.6 
        noise_clip_1 = 0.5 
        noise_clip_2 = 0.5 
        noise_1 = jnp.clip((jax.random.normal(key, next_action.shape) * policy_noise_1), -noise_clip_1, noise_clip_1) 
        noise_2 = jnp.clip((jax.random.normal(key, batch.actions.shape) * policy_noise_2), -noise_clip_2, noise_clip_2) 
        next_action = jnp.clip(next_action + noise_1, -1, 1) #add noise to next_action
        ood_action = jnp.clip(batch.actions + noise_2, -1, 1) #add noise to batch_actions --> ood action
        # compute targetQ
        key, next_action = get_action(
            key, actor_state, batch.next_observations, temperature=0.0)
        target_q1, target_q2 = qf.apply(
            qf_state.target_params, batch.next_observations, next_action)
        target_q = batch.rewards + args.gamma * \
            batch.masks * (jnp.minimum(target_q1, target_q2)) 
        q1_off, q2_off = qf.apply(qf_state.params, batch.observations, batch.actions)

        def qf_loss_fn(params):
            curr_target_q1, curr_target_q2 = qf.apply(params, batch.observations, ood_action)
            beta = args.beta
            q1, q2 = qf.apply(params, batch.observations, batch.actions) #batch.actions(256,6), q1(256,1)
            y1 = target_q
            qf_loss = ((q1 - y1)**2 + (q2 - y1)**2).mean() + beta * ((curr_target_q1 - q1_off)**2 + (curr_target_q2 - q2_off)**2).mean()

            return qf_loss, {
                "qf_loss": qf_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
            }

        (qf_loss, info), grads = jax.value_and_grad(
            qf_loss_fn, has_aux=True)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)
        return qf_state, info

    # soft update for target_network (common use) both actor and critic
    def update_actor_target(actor_state):
        new_target_params = jax.tree_map(
            lambda p, tp: p * args.polyak + tp *
            (1 - args.polyak), actor_state.params,
            actor_state.target_params)
        return actor_state.replace(target_params=new_target_params)
    
    def update_target(qf_state):
        new_target_params = jax.tree_map(
            lambda p, tp: p * args.polyak + tp *
            (1 - args.polyak), qf_state.params,
            qf_state.target_params)
        return qf_state.replace(target_params=new_target_params)

    # update the network parameters
    @jax.jit
    def update1(qf_state, batch, key): 
        qf_state, qf_info = update_qf(
            actor_state, qf_state, batch, key)
        return qf_state, {**qf_info}
    
    @jax.jit
    def update2(actor_state, qf_state, batch, key): 
        actor_state, actor_info = update_actor(
            actor_state, qf_state, batch, key)
        qf_state, qf_info = update_qf(
            actor_state, qf_state, batch, key) 
        actor_state = update_actor_target(actor_state)
        qf_state = update_target(qf_state)
        return actor_state, qf_state, {
            **actor_info, **qf_info
        }

    # Main loop 
    for global_step in tqdm(range(args.total_iterations), desc="Training", unit="iter"):

        # Batch update
        batch = dataset.sample(batch_size=args.batch_size)
        qf_state, update_info = update1(qf_state, batch, key)
        m = 2
        if global_step % m == 0:
            actor_state, qf_state, update_info = update2(actor_state, qf_state, batch, key)

        # Evaluation
        if global_step % args.eval_freq == 0:
            env.seed(args.seed)
            stats = {"return": [], "length": []}
            for _ in range(args.eval_episodes):
                obs, done = env.reset(), False
                while not done:
                    key, action = get_action(
                        key, actor_state, obs, temperature=0.0)
                    action = np.asarray(action)
                    obs, reward, done, info = env.step(action)
                for k in stats.keys():
                    stats[k].append(info["episode"][k[0]])
            for k, v in stats.items():
                writer.add_scalar(
                    f"charts/episodic_{k}", np.mean(v), global_step)
                if k == "return":
                    normalized_score = env.get_normalized_score(
                        np.mean(v)) * 100
                    writer.add_scalar("charts/normalized_score",
                                      normalized_score, global_step)
            writer.flush()

        # Logging
        if global_step % args.log_freq == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    writer.add_scalar(f"losses/{k}", v, global_step)
                else:
                    writer.add_histogram(f"losses/{k}", v, global_step)
            writer.add_scalar("charts/SPS", int(global_step /
                              (time.time() - start_time)), global_step)
            writer.flush()

    env.close()
    writer.close()
