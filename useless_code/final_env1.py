'''
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

'''

import numpy as np
import time
import math
import sys

from numpy.core._multiarray_umath import ndarray

'''
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
'''

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

# state shape= (21,1)   第一位是已选择导弹的数量，后20位是导弹的选择目标
# action shape=(7,1)   可选择的目标
# 不带角度
class Select():

    def __init__(self):
        # 更新
        target1_1 = Boat(x=400, y=50, r=0, passby=0, defend=0, v=300, die_v=300, name='T1_1')
        target2_1 = Boat(x=350, y=20, r=20, passby=0.4, defend=0.6, v=50, die_v=50, name='T2_1')    # 敢死队前上
        target2_2 = Boat(x=350, y=80, r=20, passby=0.4, defend=0.6, v=50, die_v=50, name='T2_2')    # 敢死队前下
        target2_3 = Boat(x=375, y=15, r=20, passby=0.4, defend=0.6, v=50, die_v=50, name='T2_3')    # 敢死队后上
        target2_4 = Boat(x=375, y=85, r=20, passby=0.4, defend=0.6, v=50, die_v=50, name='T2_4')     # 敢死队后下
        target2_5 = Boat(x=390, y=50, r=12, passby=0.3, defend=0.5, v=50, die_v=50, name='T2_5')    # 航母前
        target2_6 = Boat(x=400, y=40, r=12, passby=0.3, defend=0.5, v=50, die_v=50, name='T2_6')    # 航母上
        target2_7 = Boat(x=410, y=50, r=12, passby=0.3, defend=0.5, v=50, die_v=50, name='T2_7')    # 航母后
        target2_8 = Boat(x=400, y=60, r=12, passby=0.3, defend=0.5, v=50, die_v=50, name='T2_8')    # 航母下

        # todo  这里需要根据地图的具体大小进行更改
        missile1 = Missile(x=220, y=14, type=0, name='Z1')   # 导弹1
        missile2 = Missile(x=220, y=22, type=0, name='Z2')   # 导弹2
        missile3 = Missile(x=220, y=30, type=0, name='Z3')   # 导弹3
        missile4 = Missile(x=220, y=38, type=0, name='Z4')   # 导弹4
        missile5 = Missile(x=220, y=46, type=0, name='Z5')   # 导弹5
        missile6 = Missile(x=220, y=54, type=0, name='Z6')   # 导弹6
        missile7 = Missile(x=220, y=62, type=0, name='Z7')   # 导弹7
        missile8 = Missile(x=220, y=70, type=0, name='Z8')   # 导弹8
        missile9 = Missile(x=220, y=78, type=0, name='Z9')   # 导弹9
        missile10 = Missile(x=220, y=86, type=0, name='Z10')   # 导弹10
        missile11 = Missile(x=180, y=14, type=0, name='Z11')   # 导弹11
        missile12 = Missile(x=180, y=22, type=0, name='Z12')   # 导弹12
        missile13 = Missile(x=180, y=30, type=0, name='Z13')   # 导弹13
        missile14 = Missile(x=180, y=38, type=0, name='Z14')   # 导弹14
        missile15 = Missile(x=180, y=46, type=0, name='Z15')   # 导弹15
        missile16 = Missile(x=180, y=54, type=0, name='Z16')   # 导弹16
        missile17 = Missile(x=180, y=62, type=0, name='Z17')   # 导弹17
        missile18 = Missile(x=180, y=70, type=0, name='Z18')   # 导弹18
        missile19 = Missile(x=180, y=78, type=0, name='Z19')   # 导弹19
        missile20 = Missile(x=180, y=86, type=0, name='Z20')   # 导弹20

        self.gamma_r = 0.5    # 折扣因子,导弹被打下来的时候对回报进行折扣
        self.belta_V = 0.3    # 折扣因子,舰艇被打百分之多少后认为死亡
        self.punish_value_r = 2  # 惩罚边界,当r_out小于这个数时进行惩罚

        self.action_space = [target1_1, target2_1, target2_2, target2_3, target2_4, target2_5,
                             target2_6, target2_7, target2_8]   # 目标选择
        self.missile_list = [missile1, missile2, missile3, missile4, missile5, missile6, missile7,
                             missile8, missile9, missile10, missile11, missile12, missile13, missile14,
                             missile15, missile16, missile17, missile18, missile19, missile20]
        self.n_action = len(self.action_space)  # 可选动作数量（根据状态不同而变化）
        self.n_agent_type = 1  # 行为体种类数量
        self.n_agent = len(self.missile_list)

        self.distance_matrix = np.zeros([self.n_action, self.n_agent], dtype=float)
        # 加入舰艇和导弹距离的千分之一的数值作为小权重，相对距离越远导弹命中的概率越低
        for i in range(len(self.distance_matrix)):
            for j in range(len(self.distance_matrix[0])):
                self.distance_matrix[i][j] = int(self.action_space[i].get_distance(
                    self.missile_list[j], self.action_space[i]))/10000

        self.pair = [0, 0]    # len=2, index0=missile, index1=target
        self.title = 'Target Select'
        self.count = 0
        self.V = np.array([[target.V for target in self.action_space]])   # 多一层中括号，为了维度对应
        self.A = self.V
        self.array_r = np.log(self.V)
        self.l_table = np.array([[0.6], [0.9]])
        # self.l_table = np.array([[0.6, 1],
        #                         [0.9, 2]])    # 这里定义好了两类导弹,可扩展性不好
        self.num_z = np.zeros((1, self.n_agent_type))  # 各种类行为体数量
        self.cost_value = np.array([0.8, 1.0])  # 代价函数权重   # 这里也是可扩展性的问题
        self.Eta = 0  # 综合效费比

        self.state = (-1)*np.ones(self.n_agent)
        self.r = 0
        self.cost = 0  # 代价函数初始化
        self.n_feature = len(self.state)  # 状态量维数
        self.s_index = 0
        self.target_alive = np.ones((1, self.n_action))   # 舰艇的存活情况
        # self.r_total = 0


    def reset(self):
        self.state = (-1)*np.ones(self.n_agent)
        self.V = np.array([[target.V for target in self.action_space]])
        self.A = self.V
        self.array_r = np.log(self.V)
        self.r = 0
        self.Eta = 0
        self.count = 0  # 已分配导弹的计数器
        self.cost = 0  # 代价函数初始化
        self.num_z = np.zeros((1, self.n_agent_type))  # 各种类行为体数量
        # self.r_total = 0
        self.s_index = 0    # 即将分配的导弹的索引
        self.target_alive = np.ones(self.n_action)   # 舰艇的存活情况
        for i in self.action_space:
            i.alive = True    # 血月来临~万物复活
            i.total_loss = 0

        return self.state

    def step(self, action):
        for i, alive in np.ndenumerate(self.target_alive):
            if alive == 0:
                self.action_space[i[0]].alive = False
        s_ = self.state   # 初始化状态为step前的状态
        missile = self.missile_list[np.int(self.s_index)]  # 指定行为体类型

        self.pair[0] = missile  # 添加missile
        self.num_z[0, missile.type] += 1  # 更新各类型行为体数量

        # if self.count < self.n_agent:
        #     if action in range(self.n_action):
        s_[self.s_index] = action
        self.s_index += 1

        target = self.action_space[action]  # action对应的对象
        self.pair[1] = target    # 添加target
        print(self.pair[0].name+': '+str(self.pair[0].type)+' --> '+self.pair[1].name)
        self.count += 1

        # 奖励函数
        if target in self.action_space:  # 若行为体类型和目标类型已选取完毕
            alive_count = 0    # 存活舰艇的数量,用于判断是否done
            avoid_hit_probability = 1
            # 更新
            for t in self.action_space:
                if t.alive:
                    alive_count += 1    # 记录存活舰艇的数量
                    if t == target:
                        avoid_hit_probability *= t.not_defend_probability()
                    else:
                        avoid_hit_probability *= t.not_hitpass_probability(missile, target)

            # 更新
            # 距离越远命中概率越低
            avoid_hit_probability -= self.distance_matrix[action][self.s_index-1]

            # 采用期望模型,无论是否打中舰艇l值都会增加
            reward = self.get_reward(missile, target, action)  # 得到的是累计奖励
            # 回报乘以概率的折扣
            r_out = (reward-self.r)*avoid_hit_probability
            target.total_loss += r_out
            # 如果目标死亡,给一个折扣;如果回报太低，给一个惩罚，让导弹倾向于更换目标
            if not target.alive:
                r_out *= self.gamma_r
            if r_out < self.punish_value_r:
                r_out = -10  # (reward-self.r)*(-0.2)
            print('reward: ' + str(reward))
            print('self.r: ' + str(self.r))
            print('r_out_pre: ' + str(reward-self.r))
            print('我躲避舰艇的概率: '+str(avoid_hit_probability))
            print('r_out: ' + str(r_out))
            print('target.total_loss: ' + str(target.total_loss))
            print('target.V*0.3: ' + str(target.V * self.belta_V))
            # 判断舰艇是否死亡,如果死亡就等for循环后更新
            if target.total_loss > target.V*self.belta_V:
                # target.alive = False
                self.target_alive[action] = 0
                alive_count -= 1
                print('I hit the target! His total_loss is: ' + str(target.total_loss))
            # reward = self.r + cur_reward
            # 判断是否terminal
            if self.count == self.n_agent or alive_count == 0:  # 如果进行了20次目标选择
                done = 1
                # print('Done')
            else:
                done = 0
        else:
            print("galigaygay")
            print(target.name)
            done = 0

        self.state = s_
        # r_out = (reward - self.r)
        self.r = reward
        #self.r_total += r_out
        return s_, r_out, done  # 返回下一步的状态 s_,奖励值 reward,完成标志 done
        # return s_, r_sum, done  # 返回下一步的状态 s_,奖励值 reward,完成标志 done

    # 奖励函数
    def get_reward(self, missile, target, action):
        l_col = missile.type
        # todo  这里需要确定哪里是航母舰
        # 更新
        if target.name == 'T1_1':  # 若目标类型为航母
            l_row = 0
        else:
            l_row = 1
        l = self.l_table[l_row, l_col]  # 根据行为体和目标类型读取l值
        r = np.zeros((1, self.n_action))
        r[0, action] = l
        # 这里可能会有问题，mark一下
        self.array_r = np.subtract(self.array_r, r)
        self.A = np.subtract(self.V, np.exp(self.array_r))  # 计算衰减值A
        # 这里cost没用到
        cost = 0
        for i in range(0, self.n_agent_type):
            cost += (self.num_z[0, np.int(i)] * self.cost_value[np.int(i)])
        # reward = np.sum(self.A) /cost # 将各目标衰减值求和作为奖励值
        reward = np.sum(self.A)
        return reward

    def print_target(self, count_target):
        length = len(count_target)
        # 导弹击中舰艇的概率列表
        hit_probability = [[1]*self.n_agent for _ in range(self.n_action)]
        # 更新
        # 计算每个导弹打每个舰艇的击中概率
        for i in range(self.n_action):   # target
            for j in range(self.n_agent):   # missile
                for t in self.action_space:
                    if t.alive:
                        if t == self.action_space[i]:
                            hit_probability[i][j] *= t.not_defend_probability()
                        else:
                            hit_probability[i][j] *= t.not_hitpass_probability(self.missile_list[j], self.action_space[i])

        # 距离越远命中概率越低
        for i in range(len(hit_probability)):
            for j in range(len(hit_probability[0])):
                hit_probability[i][j] -= self.distance_matrix[i][j]

        choosen_agent = set()    # 已经选择的导弹编号存起来
        agent_result = [[] for _ in range(self.n_action)]    # 收集每个舰艇的分配结果
        for i in range(length-1, -1, -1):   # 先给后面的舰艇分配导弹
            argarray = np.argsort(hit_probability[i])[::-1]   # 最有可能击中该舰艇的导弹由大到小排序
            print("舰艇："+str(i))
            print("概率：", end="")
            print(argarray)
            count = 0
            index = 0
            while count < count_target[i]:
                if argarray[index] not in choosen_agent:
                    choosen_agent.add(argarray[index])
                    agent_result[i].append(argarray[index])
                    count += 1
                index += 1
        # 打印结果
        for i in range(len(agent_result)):
            print(agent_result[i])
            # for j in agent_result[i]:
            #     print(hit_probability[i][j], end="")
            #     print()
        for i in range(len(hit_probability)):   # target
            for j in range(len(hit_probability[0])):    # missile
                print(round(hit_probability[i][j], 5), end=" ")
            print()
        print(self.target_alive)
        return agent_result

    def render(self):
        time.sleep(0.1)
        self.update()


class Boat:
    def __init__(self, x, y, r, passby, defend, v, die_v, name, total_loss=0, alive=True, attack=0):
        self.x = x                      # 护卫舰横坐标
        self.y = y                      # 护卫舰纵坐标
        self.r = r                      # 护卫舰护卫半径
        self.prob_pass = passby         # 拦截经过导弹的概率
        self.prob_defend = defend       # 防御进攻导弹的概率
        self.V = v                      # 导弹打中存活舰艇的收益
        self.die_V = die_v              # 导弹打中死亡舰艇的收益(主要针对DQN里的random策略)
        self.name = name                # 命名
        self.total_loss = total_loss    # 被攻击几次后死亡
        self.alive = alive              # 是否存活
        self.attack = attack            # 护卫舰攻击力(暂未用到)

    # 计算导弹发射位置与舰艇之间的距离
    def get_distance(self, missile, target):
        x1, y1 = missile.x, missile.y
        x2, y2 = target.x, target.y
        distance = math.pow(math.pow(x1-x2, 2)+math.pow(y1-y2, 2), 0.5)
        return distance

    # 判断导弹的直线轨迹是否在护卫舰的作用范围
    def is_inside(self, missile, target):
        x1, y1 = missile.x, missile.y
        x2, y2 = target.x, target.y
        px, py = self.x, self.y
        # target就是护卫舰
        if px == x2 and py == y2:
            return True
        r = self.r
        if x1 == x2:
            a = 1
            b = 0
            c = -x1  # 特殊情况判断，分母不能为零
        elif y1 == y2:
            a = 0
            b = 1
            c = -y1  # 特殊情况判断，分母不能为零
        else:
            a = y1-y2
            b = x2-x1
            c = x1*y2 - y1*x2
        # d = |Ax0+By0+c|/sqrt(A^2+B^2)
        dist1 = a*px + b*py + c
        dist1 *= dist1
        dist1 = dist1 / (a*a + b*b)
        dist2 = r*r
        # 点到直线距离大于半径r,在范围外.如果两者相等就相当于在范围外
        if dist1 >= dist2:
            return False
        angle1 = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
        angle2 = (px - x2) * (x1 - x2) + (py - y2) * (y1 - y2)
        # 其余两角余弦都为正，则是锐角(排除线段延长线在圆内的情况)
        # 线段target端点刚好位于圆上,这种情况也视为在范围外
        if angle1 > 0 and angle2 > 0:
            return True
        return False

    # 更新
    # 舰艇未能拦截经过的导弹的概率
    def not_hitpass_probability(self, missile, target):
        # num = random.random()
        if Boat.is_inside(self, missile, target):
            return 1-self.prob_pass  # 未成功拦截导弹的概率
        else:
            return 1

    # 舰艇未能成功防御朝自己进攻的导弹的概率
    def not_defend_probability(self):
        return 1-self.prob_defend

    # 计算护卫舰能拦截导弹的概率
    def hit_probability(self, missile, target):
        # num = random.random()
        if Boat.is_inside(self, missile, target):
            return self.probability   # 未成功拦截导弹的概率
        else:
            return 0

class Missile:
    def __init__(self, x, y, type, name, attack=0):
        self.x = x                  # 导弹发射位置横坐标
        self.y = y                  # 导弹发射位置纵坐标
        self.type = type            # 导弹类型
        self.name = name            # 导弹名字
        self.attack = attack        # 导弹攻击力(暂未用到)




