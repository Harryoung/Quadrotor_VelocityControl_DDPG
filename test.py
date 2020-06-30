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
from quadrotorsim import QuadrotorSim
from env import Quadrotor

# disable gpu
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
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            if obs.shape[0] == 19:
                # yaw = obs[14]
                # pitch = obs[12]
                # roll = obs[13]
                next_target_g_v_x = obs[16]
                next_target_g_v_y = obs[17]
                next_target_g_v_z = obs[18]
                # r_matrix = get_rotation_matrix(yaw, pitch, roll)
                r_matrix = env.simulator.get_coordination_converter_to_body()
                next_expected_v = np.squeeze(np.matmul(r_matrix, np.array(
                    [[next_target_g_v_x], [next_target_g_v_y], [next_target_g_v_z]], dtype="float32")))
                obs = np.append(obs, next_expected_v)  # extend the obs
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

            # yaw = next_obs[14]
            # pitch = next_obs[12]
            # roll = next_obs[13]
            next_target_g_v_x = next_obs[16]
            next_target_g_v_y = next_obs[17]
            next_target_g_v_z = next_obs[18]
            # r_matrix = get_rotation_matrix(yaw, pitch, roll)
            r_matrix = env.simulator.get_coordination_converter_to_body()
            next_expected_v = np.squeeze(np.matmul(r_matrix, np.array(
                [[next_target_g_v_x], [next_target_g_v_y], [next_target_g_v_z]], dtype="float32")))
            next_obs = np.append(next_obs, next_expected_v)  # extend the obs

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

    return np.linalg.inv(r_matrix)


def get_action_lst(env):
    obs = env.reset()
    b_v_x_0 = obs[0]
    b_v_y_0 = obs[1]
    b_v_z_0 = obs[2]
    next_v_x_0 = obs[16]
    next_v_y_0 = obs[17]
    next_v_z_0 = obs[18]

    simulator = QuadrotorSim()
    simulator.get_config('config.json')
    simulator.reset()

    sim_v_g = simulator.global_velocity
    sim_v_b = np.matmul(simulator._coordination_converter_to_body, simulator.global_velocity)

    velocity_lst = []
    action_lst = []
    np.random.seed(0)
    for _ in range(1000):
        act = np.random.uniform(
            low=0.10, high=15.0, size=4)
        act = act.astype(np.float32)
        action_lst.append(act)
        simulator.step(act.tolist(), 0.01)

        next_obs, reward, done, info = env.step(np.array([0.1,0.1,0.1,0.1], dtype='float32'))

        b_v_x = info['b_v_x']
        b_v_y = info['b_v_y']
        b_v_z = info['b_v_z']
        next_v_x = info['next_target_g_v_x']
        next_v_y = info['next_target_g_v_y']
        next_v_z = info['next_target_g_v_z']

        # body_velocity = np.matmul(
        #         #     simulator._coordination_converter_to_body,
        #         #     simulator.global_velocity)
        #         # velocity_lst.append(list(body_velocity))
        g_v = 1 * simulator.global_velocity
        velocity_lst.append(g_v)
    return action_lst


def expected_behave(env, action_lst):
    # evaluate with action_lst
    eval_reward = []
    for i in range(2):
        env.reset()
        total_reward, steps = 0, 0
        index = 0
        while True:
            action = action_lst[index]
            index += 1
            next_obs, reward, done, info = env.step(action)
            next_expected_v_x = info['next_target_g_v_x']
            next_expected_v_y = info['next_target_g_v_y']
            next_expected_v_z = info['next_target_g_v_z']
            b_v_x = info['b_v_x']
            b_v_y = info['b_v_y']
            b_v_z = info['b_v_z']
            # next_v_x = velocity_lst[index][0]
            # next_v_y = velocity_lst[index][1]
            # next_v_z = velocity_lst[index][2]
            total_reward += reward
            steps += 1

            if done:
                break
            env.render()
        eval_reward.append(total_reward)
    return np.mean(eval_reward)

if __name__ == "__main__":
    # debug and get the expected standard answer.
    # env1 = Quadrotor(task="velocity_control", seed=0)
    # env2 = Quadrotor(task="velocity_control", seed=0)
    # action_lst = get_action_lst(env1)
    # reward = expected_behave(env2, action_lst)
    # print('best reward: {}'.format(reward))
    # exit()
    # 创建飞行器环境
    # env = make_env("Quadrotor", task="velocity_control", seed=0)
    env = Quadrotor(task="velocity_control", seed=0)
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 根据parl框架构建agent

    model = QuadrotorModel(act_dim=act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = QuadrotorAgent(algorithm=algorithm, obs_dim=obs_dim + 3, act_dim=act_dim)

    ckpt = 'model_dir/best.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
    agent.restore(ckpt)
    evaluate_reward = evaluate(env, agent)
    logger.info('Evaluate reward: {}'.format(evaluate_reward))  # 打印评估的reward
