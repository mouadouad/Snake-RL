import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver
import os
import snake

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

fc_layer_params = (100,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 10  # @param {type:"integer"}
min_q_value = 0  # @param {type:"integer"}
max_q_value = 500  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

num_eval_episodes = 2  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

train_py_env =snake.SnakeEnv()
eval_py_env = snake.SnakeEnv()

import time
now = time.time()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

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


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

compute_avg_return(eval_env, random_policy, num_eval_episodes)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)
with tf.device('/device:GPU:0'):
  for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy)

  # This loop is so common in RL, that we provide standard implementations of
  # these. For more details see the drivers module.

  # Dataset generates trajectories with shape [BxTx...] where
  # T = n_step_update + 1.
  dataset = replay_buffer.as_dataset(
      num_parallel_calls=3, sample_batch_size=batch_size,
      num_steps=n_step_update + 1).prefetch(3)

  iterator = iter(dataset)


  # (Optional) Optimize by wrapping some of the code in a graph using TF function.
  agent.train = common.function(agent.train)

  # Reset the train step
  agent.train_step_counter.assign(0)

  # Evaluate the agent's policy once before training.
  avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
  returns = [avg_return]


  for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
      collect_step(train_env, agent.collect_policy)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)

    step = agent.train_step_counter.numpy()

    if step % eval_interval == 0:
      print('step = {0}: loss = {1}   : time  =  {2:.2f} s'.format(step, train_loss.loss,time.time()-now))
      now = time.time()

    if step % eval_interval == 0:
      avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
      print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
      returns.append(avg_return)

  collect_policy = agent.collect_policy

tf_policy_saver = policy_saver.PolicySaver(collect_policy)
policydir = "model"
tf_policy_saver.save(policydir)
# Convert to tflite model
converter = tf.lite.TFLiteConverter.from_saved_model(
    policydir, signature_keys=['action'])
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
tflite_policy = converter.convert()
with open(os.path.join('snake1_tf_agents.tflite'), 'wb') as f:
    f.write(tflite_policy)