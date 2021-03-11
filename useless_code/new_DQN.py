'''
    智能体行为决策
    Deep Q Network
    learning_rate : 学习效率  α
    reward_decay ： 奖励衰减值 γ
    e_greedy     : 贪婪率 ε
    replace_target_iter : 目标网络（target net）更新间隔步数
    memory_size : 记忆库(store_transition)记忆数量
    batch_size : 网络更新梯度下降的批尺寸（单次训练样本数）
    e_greedy_increment : ε变化量(根据Q值情况增大或减小,调整探索程度)
    output_graph : 输出图

    n_actions : 输入动作(action)数量
    n_features: 输入观测量(observation)数量
    -----------------------------------------------------------
    tf.layers.Dense：密集连接(Densely-connected)层类.
    实现操作: outputs = activation(inputs * kernel + bias)
             activactivation: 激活函数（默认:None）
             kernel: 由层（layers）创建的权重矩阵
             bias: 由层（layers）创建的偏差向量

    tf.nn.relu: 神经网络激活函数.
                计算校正线性：max(features, 0)

    tf.reduce_max: 计算一个张量的各个维度上元素的最大值.
                   按照axis给定的维度减少input_tensor

    tf.stop_gradient: 停止梯度计算,可将计算图中的节点(op)转换为常量.

    tf.stack： 数组叠加 axis=0:沿x轴 axis=1:沿y轴

    tf.shape: 输出张量维度

    tf.gather_nd(params, indices, name=None)：
            用indices从张量params得到新张量
            indices = [[0, 0], [1, 1]]
            params = [['a', 'b'], ['c', 'd']]
            output = ['a', 'd']

    tf.reduce_mean: 计算张量各个维度上的元素的平均值

    tf.train.RMSPropOptimizer: 使用RMSProp(Root Mean Square Prop)算法进行优化
        一种自适应学习率方法


'''

import numpy as np
import tensorflow as tf


# 指定随机数种子（指定随机数生成起始位置）
np.random.seed(3)
tf.set_random_seed(3)


class DeepQNetwork:
    def __init__(self, n_actions, n_features, n_input1, n_input2, n_input3, n_input4, # n_input5,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy_max=0.99,
                 replace_target_iter=300,
                 memory_size=1000,
                 batch_size=128,
                 e_greedy_increment=0.0001,
                 output_graph=False,
                 prioritized=True
                 ):
        self.n_actions = n_actions  # action数量
        self.n_features = n_features  # observation数量
        # add
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_input3 = n_input3
        self.n_input4 = n_input4
        # self.n_input5 = n_input5
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy_max  # 设置ε最大值
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size  # 记忆值上限
        self.batch_size = batch_size  # 更新时从memory中提取的记忆数量
        self.epsilon_increment = e_greedy_increment  # ε增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 探索模式:若e_greedy_increment不为None,初始epsilon = 0;否则epsilon = epsilon_max.
        #优先回放
        self.prioritized = prioritized

        # 学习步数
        self.learn_step_counter = 0

        # 将记忆张量初始化（设为0张量）[s, a, r, s_]
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 3))  # 行数:memory_size 列数:[s, a, r, done, s_]维数,状态量维数*2+动作维数+奖励值维数+done

        # 构建目标网络,估计网络[target_net, evaulate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')  # 提取target_net参数
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')  # 提取eval_net参数

        # 替换 target net 的参数
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()  # 定义会话(Session)

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)  # 输出graph

        self.sess.run(tf.global_variables_initializer())  # 变量初始化
        self.cost_his = []  # 记录cost历史


    def _build_net(self):
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        # -----------------设置输入 placeholder -------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 输入当前状态s
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 输入下一步状态s_
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # 输入奖励值r
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # 输入动作a
        self.done = tf.placeholder(tf.float32, [None, ], name='done')  # 输入动作a
        # self.input1 = tf.placeholder(tf.float32, [None, self.n_input1], name='input1')
        # self.input1_ = tf.placeholder(tf.float32, [None, self.n_input1], name='input1_')
        # add
        self.input1 = tf.placeholder(tf.float32, [None, self.n_input1], name='input1')
        self.input2 = tf.placeholder(tf.float32, [None, self.n_input2], name='input2')
        self.input3 = tf.placeholder(tf.float32, [None, self.n_input3], name='input3')
        self.input4 = tf.placeholder(tf.float32, [None, self.n_input4], name='input4')
        # self.input5 = tf.placeholder(tf.float32, [None, self.n_input5], name='input5')
        self.input1_ = tf.placeholder(tf.float32, [None, self.n_input1], name='input1_')
        self.input2_ = tf.placeholder(tf.float32, [None, self.n_input2], name='input2_')
        self.input3_ = tf.placeholder(tf.float32, [None, self.n_input3], name='input3_')
        self.input4_ = tf.placeholder(tf.float32, [None, self.n_input4], name='input4_')
        # self.input5_ = tf.placeholder(tf.float32, [None, self.n_input5], name='input5_')

        w_initializer = tf.contrib.layers.xavier_initializer()
        # w_initializer = tf.random_normal_initializer(0., 0.3)  # 设置初始化器
        b_initializer = tf.constant_initializer(0.1)

        # ------------------构建 evaluate_net ---------------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.input1, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')  # 添加全连接层
            e2 = tf.layers.dense(self.input2, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')  # 添加全连接层
            e3 = tf.layers.dense(self.input3, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')  # 添加全连接层
            e4 = tf.layers.dense(self.input4, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e4')  # 添加全连接层
            # e5 = tf.layers.dense(self.input5, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e5')  # 添加全连接层
            e1234 = tf.concat([e1, e2, e3, e4], axis=1, name='e1234')  # 融合所有导弹状态
            e6 = tf.layers.dense(e1234, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e6')  # 添加全连接层
            self.q_eval = tf.layers.dense(e6, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')  # 添加全连接层q
            # 输入:e1; 输出维度(units):self.n_actions; 激活函数(activation):None(默认);
            # 卷积核的初始化器: w_initializer; 偏置项的初始化器: b_initializer;

        # -------------------构建 target_net ----------------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.input1_, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')  # 添加全连接层
            t2 = tf.layers.dense(self.input2_, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')  # 添加全连接层
            t3 = tf.layers.dense(self.input3_, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')  # 添加全连接层
            t4 = tf.layers.dense(self.input4_, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t4')  # 添加全连接层
            # t5 = tf.layers.dense(self.input5_, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
            #                     bias_initializer=b_initializer, name='t5')  # 添加全连接层
            t1234 = tf.concat([t1, t2, t3, t4], axis=1, name='t1234')  # 融合所有导弹状态
            t6 = tf.layers.dense(t1234, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t6')  # 添加全连接层

            self.q_next = tf.layers.dense(t6, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='tq')  # 添加全连接层t2
            # 输入:t1; 输出维度(units):self.n_actions; 激活函数(activation):None(默认);
            # 卷积核的初始化器: w_initializer; 偏置项的初始化器: b_initializer;

        with tf.variable_scope('q_target'):  # Q 现实
            q_target = self.r
            # if self.done == 0:   #没有结束的话
            q_target = q_target + self.gamma * (1-self.done) * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            # shape = (batchsize, 1)
            self.q_target = tf.stop_gradient(q_target)  # 对q_target的反向传播截断,将计算图中的节点(op)转换为常量

        with tf.variable_scope('q_eval'):  # Q 估计
            # a_indices的shape = (batchsize, 1) ,self.a就是经验库里选中的那个动作
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            # 把q_eval里n_actions个动作中只提取索引为a_indices的动作,shape = (batchsize, 1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
            # self.q_eval = batch * action_number, a_indices是batch * 1, 也就是说选择当前估计每个动作的Q值

        with tf.variable_scope('loss'):  # 计算损失函数
            if self.prioritized:
                self.abs_errors = tf.abs(self.q_target - self.q_eval_wrt_a)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval_wrt_a))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a))
                # self.loss = tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error')

        with tf.variable_scope('train'):  # 训练优化算法
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, done, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r, done], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            # 存储记忆
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0

            transition = np.hstack((s, [a, r, done], s_))  # 记录[s, a, r, s_]
            # np.hstack： 在水平方向上平铺数组进行拼接

            # 总 memory 大小是固定的, 如果超出总大小, 旧memory 就被新 memory 替换
            index = self.memory_counter % self.memory_size  # 取self.memory_counter对self.memory_size的余数
            self.memory[index, :] = transition

            self.memory_counter += 1

    def choose_action(self, observation):
        # add
        # target_state = observation[-7:]

        # 统一 observation 的 shape (1, size_of_observation)
        observation1 = observation[:self.n_input1]
        observation2 = observation[self.n_input1:(self.n_input1+self.n_input2)]
        observation3 = observation[(self.n_input1+self.n_input2):(self.n_input1+self.n_input2+self.n_input3)]
        observation4 = observation[(self.n_input1+self.n_input2+self.n_input3):(self.n_input1+self.n_input2+self.n_input3+self.n_input4)]
        # observation5 = observation[-self.n_input5:]
        observation1 = observation1[np.newaxis, :]
        observation2 = observation2[np.newaxis, :]
        observation3 = observation3[np.newaxis, :]
        observation4 = observation4[np.newaxis, :]
        # observation5 = observation5[np.newaxis, :]
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            # 这里好像没有softmax? 不需要softmax,因为要估计的是动作的Q值
            # actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            actions_value = self.sess.run(self.q_eval, feed_dict={self.input1: observation1,
                                                                  self.input2: observation2,
                                                                  self.input3: observation3,
                                                                  self.input4: observation4,
                                                                  # self.input5: observation5,
                                                                  })
            # print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
            print('random. ', end='')
        return action

    # 验证的时候用
    def evaluate(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        self.sess.run(self.target_replace_op)
        observation1 = observation[:self.n_input1]
        observation2 = observation[self.n_input1:(self.n_input1+self.n_input2)]
        observation3 = observation[(self.n_input1+self.n_input2):(self.n_input1+self.n_input2+self.n_input3)]
        observation4 = observation[(self.n_input1+self.n_input2+self.n_input3):(self.n_input1+self.n_input2+self.n_input3+self.n_input4)]
        # observation5 = observation[-self.n_input5:]
        observation1 = observation1[np.newaxis, :]
        observation2 = observation2[np.newaxis, :]
        observation3 = observation3[np.newaxis, :]
        observation4 = observation4[np.newaxis, :]
        # observation5 = observation5[np.newaxis, :]
        observation = observation[np.newaxis, :]
        # print(observation)
        # actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        actions_value = self.sess.run(self.q_eval, feed_dict={self.input1: observation1,
                                                              self.input2: observation2,
                                                              self.input3: observation3,
                                                              self.input4: observation4,
                                                              # self.input5: observation5
                                                              })
        # print(actions_value)
        action = np.argmax(actions_value)

        return action

    def learn(self):
        # 检查是否替换 target_net 参数（是否达到更新间隔步数self.replace_target_iter）
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)  # 更新t_param, e_param
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            # 从 memory 中随机抽取 batch_size 这么多记忆
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        if self.prioritized:
            _, abs_errors, cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                     feed_dict={self.input1: batch_memory[:, :self.n_input1],
                                self.input2: batch_memory[:, self.n_input1:(self.n_input1+self.n_input2)],
                                self.input3: batch_memory[:, (self.n_input1+self.n_input2):(self.n_input1+self.n_input2+self.n_input3)],
                                self.input4: batch_memory[:, (self.n_input1+self.n_input2+self.n_input3):(self.n_input1+self.n_input2+self.n_input3+self.n_input4)],
                                # self.input5: batch_memory[:, (self.n_input1+self.n_input2+self.n_input3+self.n_input4):self.n_features],
                                self.a: batch_memory[:, self.n_features],
                                self.r: batch_memory[:, self.n_features + 1],
                                self.done: batch_memory[:, self.n_features + 2],
                                self.input1_: batch_memory[:, (self.n_features+3):(self.n_features+3+self.n_input1)],
                                self.input2_: batch_memory[:, (self.n_features+3+self.n_input1):(self.n_features+3+self.n_input1+self.n_input2)],
                                self.input3_: batch_memory[:, (self.n_features+3+self.n_input1+self.n_input2):(self.n_features+3+self.n_input1+self.n_input2+self.n_input3)],
                                self.input4_: batch_memory[:, (self.n_features+3+self.n_input1+self.n_input2+self.n_input3):(self.n_features+3+self.n_input1+self.n_input2+self.n_input3+self.n_input4)],
                                # self.input5_: batch_memory[:, -self.n_input5:],
                                self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, cost, tmp1, tmp2 = self.sess.run(
                [self._train_op, self.loss, self.q_target, self.q_eval_wrt_a],
                feed_dict={
                    self.input1: batch_memory[:, :self.n_input1],
                    self.input2: batch_memory[:, self.n_input1:(self.n_input1+self.n_input2)],
                    self.input3: batch_memory[:, (self.n_input1+self.n_input2):(self.n_input1+self.n_input2+self.n_input3)],
                    self.input4: batch_memory[:, (self.n_input1+self.n_input2+self.n_input3):(self.n_input1+self.n_input2+self.n_input3+self.n_input4)],
                    # self.input5: batch_memory[:, (self.n_input1+self.n_input2+self.n_input3+self.n_input4):self.n_features],
                    self.a: batch_memory[:, self.n_features],
                    self.r: batch_memory[:, self.n_features + 1],
                    self.done: batch_memory[:, self.n_features + 2],
                    self.input1_: batch_memory[:, (self.n_features+3):(self.n_features+3+self.n_input1)],
                    self.input2_: batch_memory[:, (self.n_features+3+self.n_input1):(self.n_features+3+self.n_input1+self.n_input2)],
                    self.input3_: batch_memory[:, (self.n_features+3+self.n_input1+self.n_input2):(self.n_features+3+self.n_input1+self.n_input2+self.n_input3)],
                    self.input4_: batch_memory[:, (self.n_features+3+self.n_input1+self.n_input2+self.n_input3):(self.n_features+3+self.n_input1+self.n_input2+self.n_input3+self.n_input4)],
                    # self.input5_: batch_memory[:, -self.n_input5:]
                })
        self.cost_his.append(cost)  # 记录cost误差值
        print("cost"+str(cost))
        # 增加 ε,减小探索的概率
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig('loss.png')


'''
    # ------------------构建 evaluate_net ---------------------------
    with tf.variable_scope('eval_net'):
        e1 = tf.layers.dense(self.s, 256, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='e1')  # 添加全连接层e1
        # 输入:self.s; 输出维度(units):20; 激活函数(activation):tf.nn.relu,Relu = max(x,0),将小于0的数置0;
        # 卷积核的初始化器: w_initializer; 偏置项的初始化器: b_initializer;
        e2 = tf.layers.dense(e1, 256, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='e2')
        e3 = tf.layers.dense(e2, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='e3')
        self.q_eval = tf.layers.dense(e3, self.n_actions, kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer, name='q')  # 添加全连接层q
        # self.q_eval = tf.nn.softmax(eval_output, axis=1)
        # 输入:e1; 输出维度(units):self.n_actions; 激活函数(activation):None(默认);
        # 卷积核的初始化器: w_initializer; 偏置项的初始化器: b_initializer;

    # -------------------构建 target_net ----------------------------
    with tf.variable_scope('target_net'):
        t1 = tf.layers.dense(self.s_, 256, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='t1')  # 添加全连接层t1
        # 输入:self.s_; 输出维度(units):20; 激活函数(activation):tf.nn.relu,Relu = max(x,0),将小于0的数置0;
        # 卷积核的初始化器: w_initializer; 偏置项的初始化器: b_initializer;
        t2 = tf.layers.dense(t1, 256, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='t2')
        t3 = tf.layers.dense(t2, 128, tf.nn.leaky_relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='t3')
        self.q_next = tf.layers.dense(t3, self.n_actions, kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer, name='tq')  # 添加全连接层t2
        # self.q_next = tf.nn.softmax(target_output, axis=1)
        # 输入:t1; 输出维度(units):self.n_actions; 激活函数(activation):None(默认);
        # 卷积核的初始化器: w_initializer; 偏置项的初始化器: b_initializer;
            
'''

# SumTree和Memory没看,莫凡让自己看,两个类集成的挺好的就直接拿来用吧
class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)