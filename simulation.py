import gym

import csv

import tensorflow as tf

from tf_agents.trajectories.time_step import TimeStep, StepType

from agents.Agents import RandomAgent, KeepingAgent, LoweringAgent, \
    IncreasingAgent
from gym_bertrand.envs.bertrand_env import BertrandEnv
from tf_agents.environments import tf_py_environment

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

num_iterations = 20000  # @param {type:"integer"}
initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

tf.compat.v1.enable_v2_behavior()


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


def main():
    env = gym.make('bertrand-v0')

    agents = []

    marginal_cost = 0.1
    reservation_price = 0.8

    agents.append(RandomAgent(marginal_cost))
    agents.append(RandomAgent(marginal_cost))
    agents.append(KeepingAgent(marginal_cost))
    agents.append(KeepingAgent(marginal_cost))
    agents.append(LoweringAgent(marginal_cost))
    agents.append(LoweringAgent(marginal_cost))
    agents.append(IncreasingAgent(marginal_cost))
    agents.append(IncreasingAgent(marginal_cost))

    env.custom_init(len(agents) + 1, marginal_cost, reservation_price)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    # tf_env = env

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(tf_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # it's output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()
    agents.append(agent)

    random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),
                                                    tf_env.action_spec())

    with open('actions.csv', 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        learning_agent_total_reward = 0.0
        for i_episode in range(100):
            time_step = tf_env.reset()
            action = [tf_env.action_space.sample() for x in agents]
            learning_agent_episode_reward = 0.0
            for t in range(100):
                # env.render()
                obs = []
                for o in time_step.observation:
                    if type(o) == float:
                        obs.append(o)
                    else:
                        obs.append(o[0])
                writer.writerow(obs)
                action = []
                for i in range(len(agents)):
                    if i < len(agents) - 1:
                        action.append(agents[i].react(time_step.observation[i],
                                                      time_step.reward[i]))
                    else:
                        action_step = random_policy.action(time_step)
                        action.append(action_step.action)
                        time_step = tf_env.step(action)
                        learning_agent_episode_reward += time_step.reward[i]

                time_step = tf_env.step(action)

                if time_step.is_last():
                    rews = [agent.total_reward for agent in agents[:-1]] + [
                        learning_agent_episode_reward]
                    print("Episode fisnished after {} timesteps".format(t + 1))
                    env.reset()
                    print(rews)
                    # writer.writerow(rews)
                    break
            learning_agent_total_reward += learning_agent_episode_reward
    tf_env.close()


if __name__ == '__main__':
    main()
