"""
    目标分配执行环境
    num_M : 行为体数量 = 5
    num_T : 目标数量 = 7
    num_Z : 行为体类型 = 2
             Z1 = 600kg 战斗部
             #Z2 = 1200kg 战斗部
    num_V : 目标类型 = 3 （T1 x 1, T2 x 4, T3 x 2）
             V1 = 10 航母    I类目标
             V2 = 4  驱逐舰  II类目标
             V2 = 6  驱逐舰  III类目标
    l_table : λ值表 行为体选取相应目标的评价值
                |   Z1  |  # Z2  |
        |  T1   |  0.6  |  #1.0  |
        | T2&T3 |  0.9  |  #2.0  |
    A : 评价函数
        Ai = Vi（1 - exp(-sum(l))）
    Eta : 综合效费比
        Eta = Ai/(0.8*nz1 + 1.0*nz2)
   更改: 指定行为体类型, 修改状态量编码问题
        self.state = np.zeros(7)
        self.state[0]： 已选取目标数量, 初始为0, 最大为6.
        self.state[1]~[7]: 各个步骤选取的目标, 未选为0
   更改: 只规定进攻弹总数量，不再对各战斗部类型具体数量进行规定
        0~8 : 由Z1打击目标T1~T9
        9~17 : 由Z2打击目标T1~T9
         行为体数量由6更改为8
        self.state = np.zeros(9)
        self.state[0]： 已选取目标数量, 初始为0, 最大为8.
        self.state[1]~[9]: 各个步骤选取的目标, 未选为0
         针对场景设置要求对评价函数A、l_table、num_V进行修改，增加综合消费比Eta的计算
    更改: 指定战斗部类型和数量
         更改奖励值设定方式，将单步综合效费比变化量作为奖励值
"""
import argparse
import sys
import os
import tensorflow as tf
from DQN import DeepQNetwork
from env import Select
from utils.config import load_config
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 环境超参数
parser = argparse.ArgumentParser()
parser.add_argument('--target_mode', default='all', type=str)  # carrier/ship/all 决定攻击目标
parser.add_argument('--attack_mode', default='coord', type=str)  # angle/coord 决定攻击方式
parser.add_argument('--env_num', default=1, type=int)  # 1or2 决定采用哪个环境
# RL超参数
parser.add_argument('--learning_rate', default=0.0005, type=float)  # 学习率
parser.add_argument('--reward_delay', default=0.9, type=float)  # 折扣因子
parser.add_argument('--replace_target_iter', default=200, type=int)  # target网络更新频率
parser.add_argument('--memory_size', default=200, type=int)  # 记忆库大小
parser.add_argument('--output_graph', default=True, type=bool)  # 是否输出图
parser.add_argument('--prioritized', default=False, type=bool)  # 是否优先级采样
# 训练超参数
parser.add_argument('--max_episode', default=50000, type=int)  # 最大训练次数
parser.add_argument('--learn_episode', default=1000, type=int)  # 开始学习的迭代次数
parser.add_argument('--reward_threshold', default=600, type=int)  # 训练提前终止的回报条件
parser.add_argument('--episode_threshold', default=45000, type=int)  # 训练提前终止的迭代次数
args = parser.parse_args()
# 根据环境和攻击目标加载对应配置文件
config_path = sys.path[0] + '/configs/env' + str(args.env_num) + '_hit' + args.target_mode + '.yml'
print(config_path)
config = load_config(config_path)

reward_list = []

RL = None
for n_agent in range(1, 13):
    for seed in range(1, 13):

        env = Select(params=config['env'], n_agent=n_agent, attack_mode=args.attack_mode)
        # 解决for循环模型重加载问题
        tf.reset_default_graph()
        RL = DeepQNetwork(env.n_action, env.n_agent,
                          seed=seed,
                          learning_rate=args.learning_rate,
                          reward_decay=args.reward_delay,  # 更注重短期奖励还是长期奖励
                          replace_target_iter=args.replace_target_iter,
                          memory_size=args.memory_size,
                          output_graph=args.output_graph,
                          n_input=n_agent,
                          prioritized=args.prioritized
                          )
        reward_max = 0
        observation_max = []
        episode_max = 0
        # reward_last = 0
        # observation_last = []
        # episode_last = 0
        reward = 0
        step = 0
        print('===========' + ' start train! ' + '===========')
        print()
        for episode in range(args.max_episode):
            #  观测量初始化
            observation = env.reset()
            reward_0 = 0
            if step > args.learn_episode:
                RL.learn()
            while True:

                action = RL.choose_action(observation)

                # RL算法执行动作(action)并获取下一步观测量及奖励值
                observation_, reward, done = env.step(action)  # 在step里对导弹的位置进行了更新
                step += 1
                print('reward: ' + str(reward))
                reward_0 += reward
                # 根据过程进行RL学习
                RL.store_transition(observation, action, reward, done, observation_)

                # 转换观测量
                observation = observation_

                # 当片段结束时停止while循环
                if done:
                    print(observation)
                    print('r = ', reward_0)
                    print('===========', episode, '===========')
                    reward_list.append(reward_0)
                    break
            # 记录网络训练过程中出现最大回报的时刻
            if (reward_0 > reward_max) and (episode > args.episode_threshold):
                reward_max = reward_0
                episode_max = episode
                observation_max = observation
                # target网络参数替代
                RL.sess.run(RL.target_replace_op)
                # 保存训练结果比较好的网络
                saver = tf.train.Saver()
                # model_path = sys.path[0] + '/saved_model/env' + str(args.env_num) + '/hit_' + args.target_mode + '/' \
                #              + args.attack_mode + '_attack' + '/params.ckpt'
                model_path = sys.path[0] + '/saved_model/env' + str(args.env_num) + '/hit_' + args.target_mode + \
                             '/missile_' + str(n_agent) + '/seed' + str(seed)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                model_path += '/params.ckpt'
                # print(model_path)
                saver.save(RL.sess, model_path)
    # # 记录网络训练过程中最后满足阈值停止的回报时刻, 这个没法用了, 因为不同导弹不同环境不同任务的回报都不一样
    # if (reward_0 >= args.reward_threshold) and (episode > args.episode_threshold):
    #     reward_last = reward_0
    #     observation_last = observation
    #     episode_last = episode
    #     print('train last result is:')
    #     print(observation)
    #     print(reward_0)
    #     # target网络参数替代
    #     RL.sess.run(RL.target_replace_op)
    #     # 保存训练结果比较好的网络
    #     saver = tf.train.Saver()
    #     model_path = sys.path[0] + '/saved_model/env' + str(args.env_num) + '/hit_' + args.target_mode + '/' \
    #                  + args.attack_mode + '_attack' + '/params.ckpt'
    #     print(model_path)
    #     saver.save(RL.sess, model_path)
    #     break

# print('===========' + ' start test! ' + '===========')
# print()
# for episode in range(2):
#     # 设置随机种子,这里设置的话输出结果就是一样的,不设置的话输出就是不一样的
#     # from numpy.random import seed
#     # seed(1)
#     #  观测量初始化
#     observation = env.reset()
#     # 更新
#     # 测试的时候把导弹的位置还原
#     # observation = env.reset_all()
#     reward_0 = 0
#     print(observation)
#     while True:
#         # 选择最优策略
#         action = RL.evaluate(observation)
#
#         # RL算法执行动作(action)并获取下一步观测量及奖励值
#         observation_, reward, done = env.step(action)
#         print('reward: ' + str(reward))
#         reward_0 += reward
#
#         # 转换观测量
#         observation = observation_
#         print(observation)
#         # 当片段结束时停止while循环
#         if done:
#             print('r = ', reward_0)
#             print('===========', episode, '===========')
#             print(observation)
#             # # 记录网络输出的每个导弹攻击的目标,进而得到每个舰艇挨打的次数
#             # # 然后再根据挨打的次数采用确定型模型来确定哪些攻击概率高的导弹,使这些导弹来打它
#             # agent_result = env.print_target(count_target)
#             break
#
# 结束过程
# print('Reward = ', reward)
# print('Reward Max = ', reward_max)
# print('Episode Max = ', episode_max)
# print('Observation_Max = ', observation_max)
# target_count = [[] for _ in range(env.n_action)]
# for index, target in enumerate(observation_max):
#     target_count[int(target)].append(index)
# for i in range(len(target_count)):
#     print(env.action_space[i].name + ": " + str(target_count[i]))
# print('Reward Last = ', reward_last)
# if reward_last == 0:
#     print("在训练" + str(args.episode_threshold) + "个回合后未出现回报超过" + str(args.reward_threshold) + "的攻击策略")
# else:
#     print('Reward Last = ', reward_last)
#     print('Episode Last = ', episode_last)
#     print('Observation_Last = ', observation_last)
#     target_count = [[] for _ in range(env.n_action)]
#     for index, target in enumerate(observation_last):
#         target_count[int(target)].append(index)
#     for i in range(len(target_count)):
#         print(env.action_space[i].name + ": " + str(target_count[i]))
#     print('Done!')

mpl.use('Agg')
plt.plot(np.arange(len(reward_list)), reward_list)
plt.ylabel('Reward')
plt.xlabel('training steps')
plt.savefig('reward.png')
plt.close()

RL.plot_cost()
