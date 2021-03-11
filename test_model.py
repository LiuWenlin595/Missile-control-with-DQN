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
from final_SDQN.env import Select
from final_SDQN.DQN import DeepQNetwork
from final_SDQN.utils.config import load_config

# 环境超参数
parser = argparse.ArgumentParser()
parser.add_argument('--target_mode', default='all', type=str)   # carrier/ship/all 决定攻击目标
parser.add_argument('--env_num', default=1, type=int)   # 1or2 决定采用哪个环境
parser.add_argument('--n_agent', default=6, type=int)  # 1~18 决定采用多少导弹进攻
parser.add_argument('--attack_mode', default='coord', type=str)   # angle/coord 决定攻击方式
# RL超参数
parser.add_argument('--learning_rate', default=0.0005, type=float)      # 学习率
parser.add_argument('--reward_delay', default=0.9, type=float)          # 折扣因子
parser.add_argument('--replace_target_iter', default=200, type=int)     # target网络更新频率
parser.add_argument('--memory_size', default=200, type=int)             # 记忆库大小
parser.add_argument('--output_graph', default=True, type=bool)          # 是否输出图
parser.add_argument('--prioritized', default=False, type=bool)          # 是否优先级采样
args = parser.parse_args()
# 根据环境和攻击目标加载对应配置文件
config_path = sys.path[0] + '/configs/env' + str(args.env_num) + '_hit'+args.target_mode + '.yml'
print(config_path)
config = load_config(config_path)

env = Select(params=config['env'], n_agent=args.n_agent, attack_mode=args.attack_mode)
# 解决for循环模型重加载问题
tf.reset_default_graph()
RL = DeepQNetwork(env.n_action, env.n_agent,
                  seed=0,   # 无用
                  learning_rate=args.learning_rate,
                  reward_decay=args.reward_delay,  # 更注重短期奖励还是长期奖励
                  replace_target_iter=args.replace_target_iter,
                  memory_size=args.memory_size,
                  output_graph=args.output_graph,
                  n_input=args.n_agent,
                  prioritized=args.prioritized
                  )

print('==========='+' start test! '+'===========')
print()
# 加载训练结果好的网络
saver = tf.train.Saver()
model_path = sys.path[0] + '/saved_model/env' + str(args.env_num) + '/hit_' + args.target_mode + \
             '/missile_' + str(args.n_agent) + '/params.ckpt'
saver.restore(RL.sess, model_path)
observation = []
for episode in range(2):
    #  观测量初始化
    observation = env.reset()
    reward_0 = 0
    print(observation)
    while True:
        # 选择最优策略
        action = RL.evaluate(observation)

        # RL算法执行动作(action)并获取下一步观测量及奖励值
        observation_, reward, done = env.step(action)
        reward_0 += reward

        # 转换观测量
        observation = observation_
        # 当片段结束时停止while循环
        if done:
            print('r = ', reward_0)
            print(observation)
            print('===========', episode, '===========')
            break

    target_count = [[] for _ in range(env.n_action)]
    for index, target in enumerate(observation):
        target_count[int(target)].append(index)
    for i in range(len(target_count)):
        print(env.action_space[i].name + ": " + str(target_count[i]))

# 结束过程
print('Done!')
