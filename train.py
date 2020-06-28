import os
import numpy as np
from model import QuadrotorModel
from agent import QuadrotorAgent
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from parl.algorithms import DDPG
from env import Quadrotor
from quadrotorsim import QuadrotorSim
from test import get_rotation_matrix
from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ACTOR_LR = 0.0002   # Actor网络更新的 learning rate
CRITIC_LR = 0.001   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward
# OFFSET_SCALAR = 0.5         # OFFSET电压的缩放比例
REWARD_LIST = []
V_DIFF_LIST = []
STEPS_LIST = []


def _get_velocity_diff(velocity, velocity_target):
    vt_x, vt_y, vt_z = velocity_target
    diff = abs(vt_x - velocity[0]) + \
           abs(vt_y - velocity[1]) + \
           abs(vt_z - velocity[2])
    return diff


def run_episode(env, agent, rpm, total_steps):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        if obs.shape[0] == 19:
            yaw = obs[14]
            pitch = obs[12]
            roll = obs[13]
            next_target_g_v_x = obs[16]
            next_target_g_v_y = obs[17]
            next_target_g_v_z = obs[18]
            r_matrix = get_rotation_matrix(yaw, pitch, roll)
            next_expected_v = np.squeeze(np.matmul(r_matrix, np.array(
                    [[next_target_g_v_x], [next_target_g_v_y], [next_target_g_v_z]], dtype="float32")))
            obs = np.append(obs, next_expected_v)          # extend the obs
        else:
            next_expected_v = obs[19:]

        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        # action_main = np.random.normal(action[0], 1.0)  # 添加随机扰动，增加探索性
        # action_diff = action[1:] * OFFSET_SCALAR
        # action_new = action_diff + action_main
        # action_new = np.clip(action_new, -1.0, 1.0)
        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action_real = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action_real = action_mapping(action_real, env.action_space.low[0],
                                env.action_space.high[0])
        next_obs, reward, done, info = env.step(action_real)

        next_real_v = np.array([next_obs[0], next_obs[1], next_obs[2]], dtype="float32")
        next_expected_v = np.array([obs[16], obs[17], obs[18]], dtype="float32")
        # v_diff = np.dot(next_expected_v, next_real_v)
        v_diff = _get_velocity_diff(next_real_v, next_expected_v)
        reward_new = reward - v_diff / 10.0
        # logger.info("reward: {0}, v_diff: {1}".format(reward, v_diff))
        REWARD_LIST.append(reward)
        # V_DIFF_LIST.append(-v_diff / 10.0)
        STEPS_LIST.append(total_steps + steps)

        yaw = next_obs[14]
        pitch = next_obs[12]
        roll = next_obs[13]
        next_target_g_v_x = next_obs[16]
        next_target_g_v_y = next_obs[17]
        next_target_g_v_z = next_obs[18]
        r_matrix_ypr = get_rotation_matrix(yaw, pitch, roll)
        r_matrix = env.simulator.get_coordination_converter_to_body()
        next_expected_v = np.squeeze(np.matmul(r_matrix, np.array(
            [[next_target_g_v_x], [next_target_g_v_y], [next_target_g_v_z]], dtype="float32")))
        next_obs = np.append(next_obs, next_expected_v)  # extend the obs

        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            if obs.shape[0] == 19:
                yaw = obs[14]
                pitch = obs[12]
                roll = obs[13]
                next_target_g_v_x = obs[16]
                next_target_g_v_y = obs[17]
                next_target_g_v_z = obs[18]
                r_matrix = get_rotation_matrix(yaw, pitch, roll)
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

            yaw = next_obs[14]
            pitch = next_obs[12]
            roll = next_obs[13]
            next_target_g_v_x = next_obs[16]
            next_target_g_v_y = next_obs[17]
            next_target_g_v_z = next_obs[18]
            r_matrix = get_rotation_matrix(yaw, pitch, roll)
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

if __name__ == "__main__":
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

    # parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim + 3, act_dim)

    best_test_reward = -5000
    # agent.restore('model_dir/best.ckpt')

    # 启动训练
    test_flag = 0
    total_steps = 0
    while total_steps < TRAIN_TOTAL_STEPS:
        train_reward, steps = run_episode(env, agent, rpm, total_steps)
        total_steps += steps
        # logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward

        if total_steps // TEST_EVERY_STEPS >= test_flag:  # 每隔一定step数，评估一次模型
            # plot
            plt.clf()
            # plt.plot(STEPS_LIST, V_DIFF_LIST, c='b', label='-v_diff / 10.0')
            plt.plot(STEPS_LIST, REWARD_LIST, c='r', label='Reward')
            plt.title("Reward")
            plt.xlabel("global steps")
            plt.legend()
            plt.show()

            while total_steps // TEST_EVERY_STEPS >= test_flag:
                test_flag += 1

            evaluate_reward = evaluate(env, agent)
            logger.info('Steps {}, Test reward: {}, ACTOR_LR: {}, CRITIC_LR: {}'.format(
                total_steps, evaluate_reward, ACTOR_LR, CRITIC_LR))  # 打印评估的reward
            if evaluate_reward > best_test_reward:
                logger.info('Another best evaluate reward: {}'.format(evaluate_reward))
                best_test_reward = evaluate_reward
                agent.save('model_dir/best.ckpt')               # 只保存目前最好的结果

            # # 每评估一次，就保存一次模型，以训练的step数命名
            # ckpt = 'model_dir/steps_{}.ckpt'.format(total_steps)
            # agent.save(ckpt)
