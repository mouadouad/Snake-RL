from SnakeGame.snakeGame import SnakeGame
import tensorflow as tf
import numpy as np

from tf_agents.specs import TensorSpec
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.policies import random_tf_policy, epsilon_greedy_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

num_iterations = 1000000  # @param {type:"integer"}

initial_collect_steps = 10000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-4  # @param {type:"number"}
log_interval = 1000  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

initial_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay_steps = 100000  # Adjust the decay steps as needed

# Environment
train_py_env = SnakeGame(100)
eval_py_env = SnakeGame(100)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Neural Networks
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(100, ),
)

value_net = value_network.ValueNetwork(
    train_env.observation_spec(),
    fc_layer_params=(100, ),
)

# PPO Agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = ppo_agent.PPOAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_net=actor_net,
    value_net=value_net,
    optimizer=optimizer,
    num_epochs=25,
    train_step_counter=train_step_counter,
)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

epsilon = tf.Variable(initial_epsilon, trainable=False, dtype=tf.float32)

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

epsilon_greedy = epsilon_greedy_policy.EpsilonGreedyPolicy(
    collect_policy, epsilon=epsilon)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)



def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()

    # Apply epsilon-greedy exploration
    action_step = policy.action(time_step)
    # Random exploration with probability epsilon
    if np.random.uniform() < epsilon.numpy() and policy != random_policy:
        action_step = action_step._replace(action=tf.nest.map_structure(
            lambda spec: tf.expand_dims(
                tf.random.uniform(shape=spec.shape, minval=spec.minimum, maxval=spec.maximum + 1, dtype=spec.dtype),
                axis=-1
            ),
            train_env.action_spec()
        ))

    # Add trajectory to the replay buffe
    # action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    traj["policy_info"] = {'dist_params': {'logits': TensorSpec(shape=(3,), dtype=tf.float32, name='CategoricalProjectionNetwork_logits')}}

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

    print("buffer inside: ", buffer.gather_all())


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

     # Update epsilon based on the annealing schedule
    epsilon.assign(tf.maximum(final_epsilon, initial_epsilon - tf.cast(agent.train_step_counter, tf.float32) / epsilon_decay_steps))

    # Collect data from the training environment using epsilon-greedy exploration
    # collect_data(train_env, epsilon_greedy, replay_buffer, collect_steps_per_iteration)


    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss1 = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss1))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

collect_policy = agent.collect_policy


