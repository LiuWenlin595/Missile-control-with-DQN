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
    l_table : λ值表 行为体选取相应目标的评价值   #为什么航母的比护卫舰的小
                |   Z1  |  # Z2  |
        |  T1   |  0.6  |  #1.0  |
        | T2&T3 |  0.9  |  #2.0  |
    A : 评价函数
        Ai = Vi（1 - exp(-sum(l))）
    Eta : 综合效费比
        Eta = Ai/(0.8*nz1 + 1.0*nz2)
        # cost 还没有被用到
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
    更改: 调整奖励值的设定，将单步毁伤值变化量作为奖励值
"""

import numpy as np
import math
from objects import *


class Select:

    def __init__(self, n_agent, attack_mode, params=None):
        # 更新
        self.attack_mode = attack_mode
        self.angle = params['angle']  # 用来确定偏斜的角度
        self.target_mode = params['target_mode']  # 攻击目标
        self.gamma_r = params['gamma_r']  # 折扣因子,导弹被打下来的时候对回报进行折扣
        self.belta_V = params['belta_V']  # 折扣因子,舰艇被打百分之多少后认为死亡
        self.punish_value_r = params['punish_value_r']  # 惩罚边界,当r_out小于这个数时进行惩罚
        self.n_carrier = params['n_carrier']  # 航母的数量
        self.n_ship = params['n_ship']  # 舰艇的数量
        self.n_action = self.n_carrier + self.n_ship  # 可选动作数量（根据状态不同而变化）
        self.n_agent = n_agent  # 重新定义导弹数量使其可变
        # self.n_agent = params['n_missile']  # 导弹的数量
        self.n_agent_type = params['n_agent_type']  # 行为体种类数量
        self.action_space = []
        self.missile_list = []

        # 添加航母
        for i in range(3):
            try:
                carrier = Ship(params=params['carrier_{}'.format(i)])
            except KeyError:
                continue
            self.action_space.append(carrier)
        # 添加舰艇
        for i in range(15):
            try:
                ship = Ship(params=params['ship_{}'.format(i)])
            except KeyError:
                continue
            self.action_space.append(ship)
        # 添加导弹
        for i in range(1, self.n_agent + 1):
            try:
                missile = Missile(params=params['missile_{}'.format(i)])
            except KeyError:
                continue
            self.missile_list.append(missile)

        self.distance_matrix = np.zeros([self.n_action, self.n_agent], dtype=float)
        # 加入舰艇和导弹距离的千分之一的数值作为小权重，相对距离越远导弹命中的概率越低
        for i in range(len(self.distance_matrix)):
            for j in range(len(self.distance_matrix[0])):
                self.distance_matrix[i][j] = int(get_distance(self.missile_list[j], self.action_space[i])) / 10000
        # pair数组随便填个坑,避免warning
        self.pair = [self.missile_list[0], self.action_space[0]]  # len=2, index0=missile, index1=target
        self.title = 'Target Select'
        self.count = 0
        self.V = np.array([[target.V for target in self.action_space]])  # 多一层中括号，为了维度对应
        self.A = self.V
        self.array_r = np.log(self.V)
        self.l_table = np.array([[0.5365], [0.8047]])
        self.num_z = np.zeros((1, self.n_agent_type))  # 各种类行为体数量
        self.cost_value = np.array([0.8, 1.0])  # 代价函数权重   # 这里也是可扩展性的问题
        self.Eta = 0  # 综合效费比

        self.state = (-1) * np.ones(self.n_agent)
        self.r = 0
        self.cost = 0  # 代价函数初始化
        self.s_index = 0
        self.target_alive = np.ones((1, self.n_action))  # 舰艇的存活情况

    def reset(self):
        self.state = (-1) * np.ones(self.n_agent)
        self.V = np.array([[target.V for target in self.action_space]])
        self.A = self.V
        self.array_r = np.log(self.V)
        self.r = 0
        self.Eta = 0
        self.count = 0  # 已分配导弹的计数器
        self.cost = 0  # 代价函数初始化
        self.num_z = np.zeros((1, self.n_agent_type))  # 各种类行为体数量
        # self.r_total = 0
        self.s_index = 0  # 即将分配的导弹的索引
        self.target_alive = np.ones(self.n_action)  # 舰艇的存活情况
        for i in self.action_space:
            i.alive = True  # 血月来临~万物复活
            i.total_loss = 0

        return self.state

    def step(self, action):
        for i, alive in np.ndenumerate(self.target_alive):
            if alive == 0:
                # i shape: tuple (i,)
                self.action_space[i[0]].alive = False
        s_ = self.state  # 初始化状态为step前的状态
        missile = self.missile_list[np.int(self.s_index)]  # 指定行为体类型

        self.pair[0] = missile  # 添加missile
        self.num_z[0, missile.type] += 1  # 更新各类型行为体数量

        s_[self.s_index] = action
        if self.attack_mode == 'angle':
            # 如果采用角度攻击需要更新导弹的位置, 使用新的位置来计算回报
            if self.s_index < self.n_agent / 2:  # 前面一排的导弹
                self.missile_list[self.s_index].x = self.action_space[action].missile_x[0]
            else:
                self.missile_list[self.s_index].x = self.action_space[action].missile_x[1]
            bias = (self.action_space[action].x - self.missile_list[self.s_index].x) \
                   * math.tan(math.degrees(self.angle[self.s_index]))
            # 根据导弹初始y轴位置的正负来确定导弹重新定位的位置
            if self.action_space[action].y > 0:
                self.missile_list[self.s_index].y = self.action_space[action].y + bias
            elif self.action_space[action].y < 0:
                self.missile_list[self.s_index].y = self.action_space[action].y - bias
            else:  # ==0
                self.missile_list[self.s_index].y = self.action_space[action].y + \
                                                    round(np.random.uniform(-1, 1)) * bias
        elif self.attack_mode == 'coord':
            # 如果采用位置攻击则直接指定target就行了
            pass
        self.s_index += 1

        target = self.action_space[action]  # action对应的对象
        self.pair[1] = target  # 添加target
        print(self.pair[0].name + ': ' + str(self.pair[0].type) + ' --> ' + self.pair[1].name)
        self.count += 1

        # 奖励函数
        alive_count = 0  # 存活舰艇的数量,用于判断是否done
        avoid_hit_probability = 1   # 导弹避免被击中的概率
        # 更新
        for t in self.action_space:
            if t.alive:
                alive_count += 1  # 记录存活舰艇的数量
                if t == target:
                    avoid_hit_probability *= t.not_defend_probability()
                else:
                    avoid_hit_probability *= t.not_hitpass_probability(missile, target)

        # 距离越远躲避攻击的概率越低,体现速度的衰减对导弹的影响
        avoid_hit_probability -= self.distance_matrix[action][self.s_index - 1]

        # 采用期望模型,无论是否打中舰艇l值都会增加
        reward = self.get_reward(missile, target, action)  # 得到的是累计奖励
        # 回报乘以概率的折扣
        r_out = (reward - self.r) * avoid_hit_probability
        target.total_loss += r_out
        # 如果目标死亡,给一个折扣; 如果回报太低，给一个惩罚，让导弹倾向于更换目标
        if not target.alive:
            r_out *= self.gamma_r
        # if r_out < self.punish_value_r:
        #     r_out = -5  # (reward-self.r)*(-0.2)
        # print('reward: ' + str(reward))
        # print('self.r: ' + str(self.r))
        # print('r_out_pre: ' + str(reward-self.r))
        # print('我躲避舰艇的概率: '+str(avoid_hit_probability))
        # print('r_out: ' + str(r_out))
        # print('target.total_loss: ' + str(target.total_loss))
        # print('target.V*0.3: ' + str(target.V * self.belta_V))
        # 判断舰艇是否死亡,如果死亡就等for循环后更新
        if target.total_loss > target.V * self.belta_V:
            # target.alive = False
            self.target_alive[action] = 0
            alive_count -= 1
            # print('I hit the target! His total_loss is: ' + str(target.total_loss))
        # 判断是否terminal
        if self.count == self.n_agent or alive_count == 0:  # 如果进行了20次目标选择
            done = 1
            # print('Done')
        else:
            done = 0

        self.state = s_
        self.r = reward
        # self.r_total += r_out
        return s_, r_out, done  # 返回下一步的状态 s_,奖励值 reward,完成标志 done

    # 奖励函数
    def get_reward(self, missile, target, action):
        l_col = missile.type
        if target.name == 'T1_1' or target.name == 'T1-2':  # 若目标类型为航母
            l_row = 0
        else:
            l_row = 1
        l_value = self.l_table[l_row, l_col]  # 根据行为体和目标类型读取l值
        r = np.zeros((1, self.n_action))
        r[0, action] = l_value
        self.array_r = np.subtract(self.array_r, r)
        self.A = np.subtract(self.V, np.exp(self.array_r))  # 计算衰减值A
        # 这里cost没用到
        cost = 0
        for i in range(0, self.n_agent_type):
            cost += (self.num_z[0, np.int(i)] * self.cost_value[np.int(i)])
        reward = np.sum(self.A)
        return reward


# 静态方法
def get_distance(missile, target):
    x1, y1 = missile.x, missile.y
    x2, y2 = target.x, target.y
    distance = math.pow(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2), 0.5)
    return distance
