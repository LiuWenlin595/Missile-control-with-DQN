class BaseObject:
    def __init__(self, params):
        # params参数内容：x, y, static, vx, vy, reward, redius, name
        self.params = params
        self.x = params['x']
        self.y = params['y']
        self.name = params['name']


class Missile(BaseObject):
    def __init__(self, params):
        super(Missile, self).__init__(params)
        self.type = params['type']


# 航母和舰艇都用Ship
class Ship(BaseObject):
    def __init__(self, params):
        super(Ship, self).__init__(params)
        self.r = params['radius']                   # 护卫舰护卫半径
        self.prob_pass = params['passby_prob']      # 拦截经过导弹的概率
        self.prob_defend = params['defend_prob']    # 防御进攻导弹的概率
        self.V = params['value']                    # 导弹打中存活舰艇的收益
        self.die_V = params['die_value']            # 导弹打中死亡舰艇的收益(主要针对DQN里的random策略)
        self.total_loss = params['total_loss']      # 舰艇被攻击的累计损失
        self.alive = params['alive']                # 是否存活
        self.missile_x = params['missile_x']        # 不同角度下的以该舰艇为目标的导弹的位置
        self.missile_y = params['missile_y']        # 不同角度下的以该舰艇为目标的导弹的位置

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

    # 舰艇未能拦截经过的导弹的概率
    def not_hitpass_probability(self, missile, target):
        # num = random.random()
        if self.is_inside(missile, target):
            return 1-self.prob_pass  # 未成功拦截导弹的概率
        else:
            return 1

    # 舰艇未能成功防御朝自己进攻的导弹的概率
    def not_defend_probability(self):
        return 1-self.prob_defend
