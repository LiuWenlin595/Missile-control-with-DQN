海防环境的导弹目标选择任务

流程介绍：
敌方舰艇以固定阵型排列，我方18枚导弹依次选择攻击目标并以直线轨迹攻击，
攻击过程中若进入防御舰艇的防御半径内则有一定几率被拦截，舰艇被攻击一定次数后死亡，
同样的，可以根据任务需求，通过改变不同类型的舰艇的价值来调整导弹攻击的侧重点，
综上，需要合理选择攻击目标和攻击顺序使得期望伤害最大化



状态空间定义为18个导弹的目标选择，初始化为[-1]*18，每进行一个step填进去一个action(攻击目标)
动作空间定义为可供选择的舰艇数量，如果有7个舰艇则action space~[0, 6]，只能取整数

固定阵型默认两种，如下所示，可在configs文件夹中设计新的阵型
![image](https://user-images.githubusercontent.com/32588806/110739478-3c9e0580-826c-11eb-842e-01acdabdf65e.png)
![image](https://user-images.githubusercontent.com/32588806/110739530-53dcf300-826c-11eb-9864-4f963481f25d.png)

导弹默认两种攻击方式：
1. 位置攻击：18个导弹排成两排，位置固定，按照编号依次选择攻击目标
2. 角度攻击：每个导弹首先选择攻击目标，然后根据角度偏向确定自己的发射位置


文件：
train.py：训练文件，负责训练DQN生成指定环境指定任务指定导弹数量的攻击策略
test_model.py：测试文件，可以测试已训练好并保存的模型
DQN.py：强化学习算法文件，负责搭建神经网络，设计行动策略和目标策略，拟合Q值
env.py：环境文件，设计回报函数，计算概率与距离，模拟物理情景
objects.py：实体对象文件，设计导弹、舰艇和航母的属性和方法
loss.png：损失图样例 
文件夹：
configs：配置文件保存位置，定义实体对象的位置、价值等属性
logs：网络张量图保存位置，当output_graph定义为True时每次结束训练都会生成一张
saved_model：网络参数保存位置
useless_code：之前的一些代码，再次更改可能有一些参考价值
utils：工具包

训练：
运行train.py，可更改超参数设置里的环境编号、攻击模式，以及for循环里的导弹数量和网络随机种子

测试：
运行test_model.py，可更改超参数设置里的环境编号、攻击模式和导弹数量

对象：
导弹：坐标，编号，速度，
如果采用角度攻击的话，就没有初始位置，位置根据目标舰艇和导弹角度来确定
如果采用位置攻击的话，就初始化位置，从固定位置做直线
舰艇：编号，价值， 累计损失，是否存活，防御概率，拦截概率

DQN：
行动策略为epsilon-greedy，目标策略采用max操作，即标准的offpolicy DQN
target网络参数更新在update函数里进行

reward：
每个舰艇都有自己的固定价值V，在被击中后固定价值会得到衰减，奖励衰减函数为V*(1-exp(-x))
初始奖励为攻击目标的当前价值
然后计算导弹的被拦截概率，被拦截概率还要再减去一个距离概率(导弹和目标距离越远拦截概率越低，体现速度的衰减)
最后根据这个概率对奖励进行折扣得到期望奖励
同时把这个奖励计算进舰艇的损失里(当舰艇被击中三次就认为死亡)
如果舰艇已死亡，则在奖励的基础上进行一个超参数的死亡折扣
如果攻击前舰艇已死亡，则会获得攻击死亡舰艇的惩罚


输出样例及对应演示：
样例一 位置攻击, 无差别攻击
123

样例三 角度攻击，主攻航母