import os
import math

import numpy as np
from model import QuadrotorModel
from agent import QuadrotorAgent
from parl.utils import logger
from parl.utils import action_mapping  # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory  # 经验回放
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from parl.algorithms import DDPG

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward


def evaluate(env, agent):
    eval_reward = []
    for i in range(1):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.clip(action, -1.0, 1.0)
            action = np.squeeze(action)
            # action_main = action[0]
            # action_diff = action[1:] * OFFSET_SCALAR
            # action_new = action_diff + action_main
            # action_new = np.clip(action_new, -1.0, 1.0)
            action = action_mapping(action, env.action_space.low[0],
                                    env.action_space.high[0])

            next_obs, reward, done, info = env.step(action)

            yaw = info['yaw']
            pitch = info['pitch']
            roll = info['roll']

            r_matrix = get_rotation_matrix(yaw, pitch, roll)

            next_v = np.squeeze(np.matmul(r_matrix, np.array(
                [[info['next_target_g_v_x']], [info['next_target_g_v_y']], [info['next_target_g_v_z']]])))
            next_v_x, next_v_y, next_v_z = next_v
            print("next_v_x: {0}, next_v_y: {1}, next_y_z: {2}".format(next_v_x, next_v_y, next_v_z))
            # print("obs: {0}".format(obs))
            # print("next_obs: {0}, reward: {1}, info: {2}".format(next_obs, reward, info))
            print("b_v_x: {0}, b_v_y:{1}, b_v_z:{2}.".format(info['b_v_x'], info['b_v_y'], info['b_v_z']))
            b_v = np.array([info['b_v_x'], info['b_v_y'], info['b_v_z']], dtype="float32")
            print("next_target_g_v_x: {0}, next_target_g_v_y: {1}, next_target_g_v_z: {2}".format(
                info['next_target_g_v_x'], info['next_target_g_v_y'], info['next_target_g_v_z']))

            print("reward: {}".format(reward))
            print("Inner dot: {}".format(np.dot(next_v, b_v)))

            obs = next_obs

            total_reward += reward
            steps += 1

            if done:
                break
            env.render()
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


def get_rotation_matrix(yaw, pitch, roll):
    m_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]], dtype="float32")
    m_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]], dtype="float32")
    m_roll = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]], dtype="float32")
    r_matrix = np.matmul(m_roll, np.matmul(m_pitch, m_yaw))

    return r_matrix


if __name__ == "__main__":
    # 创建飞行器环境
    env = make_env("Quadrotor", task="velocity_control", seed=0)
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 根据parl框架构建agent

    model = QuadrotorModel(act_dim=act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = QuadrotorAgent(algorithm=algorithm, obs_dim=obs_dim, act_dim=act_dim)

    # parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

    ckpt = 'model_dir/best.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
    agent.restore(ckpt)
    evaluate_reward = evaluate(env, agent)
    logger.info('Evaluate reward: {}'.format(evaluate_reward))  # 打印评估的reward