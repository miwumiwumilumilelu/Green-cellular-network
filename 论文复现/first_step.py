import pulp
import numpy as np
import matplotlib.pyplot as plt
def LP(BS, Users, BL, R_bar, R_req):
    # 创建线性规划问题
    prob = pulp.LpProblem("ResourceAllocation", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", [(n, m) for n in BS for m in Users], lowBound=0, upBound=1, cat='Continuous')
    phi = pulp.LpVariable("phi", lowBound=0, cat='Continuous')
    # 添加约束 C7：如果用户 m 在基站 n 的阻塞列表中，则 x_{n,m} = 0
    for n in BS:
        for m in BL.get(n, []):
            prob += x[(n, m)] == 0, f"C7_n{n}_m{m}"
    #最大化 phi
    prob += phi, "Objective"
    # 添加约束 C1''：速率约束
    for m in Users:
        prob += pulp.lpSum(x[(n, m)] * R_bar.get((n, m), 0) for n in BS) >= R_req[m] + phi, f"C1''_m{m}"
    # 添加约束 C2'：资源分配限制
    for n in BS:
        prob += pulp.lpSum(x[(n, m)] for m in Users) <= 1, f"C2'_n{n}"
    # 求解问题
    prob.solve()
   # 检查求解状态
    status = pulp.LpStatus[prob.status]
    if status == "Infeasible":
        raise ValueError("The problem is infeasible. Please check the constraints.")
    result = {(n, m): x[(n, m)].varValue for n in BS for m in Users}
    optimal_phi = phi.varValue
    return status, result, optimal_phi
def LDP(BS, Users, R_bar, R_req):
    BL = {n: set() for n in BS}  # 初始化每个基站的黑名单为空集合
    X_prime = {n: {m: 0 for m in Users} for n in BS}  # 初始化用户关联 (X')，0 表示无关联
    omega = 1  # 可行性指示器
    # 检查输入数据是否合理
    for m in Users:
        total_rate = sum(R_bar.get((n, m), 0) for n in BS)
        if R_req[m] > total_rate:
            raise ValueError(f"User {m}'s required rate {R_req[m]} exceeds the total available rate {total_rate}.")
    while True:
        #通过求解松弛的 LP 问题获得 X
        status, X, optimal_phi = LP(BS, Users, BL, R_bar, R_req)
     # 如果问题不可行，抛出错误
        if status == "Infeasible":
            raise ValueError("The problem is infeasible. Please check the constraints.")        
        #对于每个用户 m，将其分配到最大化速率的基站 n*
        for m in Users:
            # 找到最大化速率的基站
            n_star = max(BS, key=lambda n: X[(n, m)] * R_bar.get((n, m), 0))            
            # 设置新的用户关联
            for n in BS:
                if n == n_star:
                    X[(n, m)] = 1
                else:
                    X[(n, m)] = 0
        #对每个基站BS, 检查资源分配是否超出限制
        for n in BS:
            total_rate = 0
            for m in Users:
                r_bar_nm = R_bar.get((n, m), 0)  # 获取 R_bar，如果不存在则默认为 0
                if r_bar_nm != 0: 
                    total_rate += X.get((n, m), 0) * R_req[m] / r_bar_nm
            #如果资源分配超过 1，则标记 omega = 0
            if total_rate > 1:
                omega = 0
                #找到新用户
                NUn = {m for m in Users if X.get((n, m), 0) == 1 and X_prime[n][m] == 0}
                #找到速率需求最高的用户加入黑名单
                if NUn:
                    m_star = max(NUn, key=lambda m: X.get((n, m), 0) * R_req[m] / R_bar.get((n, m), 1))
                    BL[n].add(m_star)
            else:
                #没有用户被加入黑名单
                omega = 1
                #如果 omega = 1，跳出循环
        if omega == 1:
            break
        else:
            omega = 1  # 重置 omega 为下一次迭代    
    #根据公式 (10) 计算资源分配 Y
    Y = {}
    for n in BS:
        for m in Users:
            if X[(n, m)] == 1:
                part1 = R_req[m] / R_bar.get((n, m), 1)
                part2 = 1 / R_bar.get((n, m), 1)
                sum_term = sum(X[(n, m_prime)] / R_bar.get((n, m_prime), 1) for m_prime in Users)
                sum_rate = sum(X[(n, m_prime)] * R_req[m_prime] / R_bar.get((n, m_prime), 1) for m_prime in Users)
                part3 = 1 - sum_rate
                Y[(n, m)] = X[(n, m)] * (part1 + part2 * (part3 / (sum_term if sum_term != 0 else 1)))
    
    return X, BL, Y
def IPC(Bs, Users, x, y, R_req, g, P_max, rho, W, eta, epsilon2=1e-6, max_iter=1000):
    print("开始IPC")
    ## 输入验证
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0") 
    N = len(Bs)
    M = len(Users)
    #初始化发射功率
    P = np.zeros(N)
    for n in range(N):
        P[n] = P_max[n] 
    def calculate_I(P):
        I = np.zeros((N, M))
        for n in range(N):
            for m in range(M):
                I[n, m] = sum(P[n_prime] * g[n_prime, m] for n_prime in range(N) if n_prime != n) + eta[m]
        print(f"干扰函数I: {I}")
        return I
    def f(n, m, P, I):
        if x[n, m] == 1:
            power_demand = (I[n, m] / g[n, m]) * (2 ** (R_req[m] / (W * y[n, m])) - 1)
            print(f"Power demand for BS {n}, User {m}: {power_demand}")
            return power_demand
        else:
            return 0
    def T(n, P, I):
        max_power_demand = max(f(n, m, P, I) for m in range(M))
        print(f"Maximum power demand for BS {n}: {max_power_demand}")
        return max_power_demand
    for k in range(max_iter):
        print(f"Iteration {k}:")
        P_old = P.copy()  #保存上一轮的功率
        I = calculate_I(P)  
        #更新每个基站的发射功率
        for n in range(N):
            P[n] = min(T(n, P, I), P_max[n])  #限制到 P_max
        #计算总功耗
        PG_old = sum(rho[n] * P_old[n] for n in range(N))
        PG_new = sum(rho[n] * P[n] for n in range(N))
        print(f"PG_old: {PG_old}, PG_new: {PG_new}, Difference: {abs(PG_new - PG_old)}")
        # 检查收敛条件
        if abs(PG_new - PG_old) < epsilon2:
            print("Converged.")
            break
    print(f"Final P: {P}")
    return P
def ECP(E, Pt, Pc, rho, alpha_n_a, alpha_a_n, max_storage=1000):
    #输入验证
    if len(E) != len(Pt) or len(E) != len(Pc):
        raise ValueError("输入参数长度不一致")
    if any(e < 0 for e in E) or any(p < 0 for p in Pt):
        raise ValueError("输入参数包含非法值（如负数）")
    # 将输入转换为 NumPy 数组
    E = np.array(E, dtype=float)
    Pt = np.array(Pt, dtype=float)
    Pc = np.array(Pc, dtype=float)
    rho = np.array(rho, dtype=float)
    alpha_n_a = np.array(alpha_n_a, dtype=float)
    alpha_a_n = np.array(alpha_a_n, dtype=float)
    #将基站分为两组
    energy_required = rho * Pt + Pc
    B1 = np.where(energy_required < E)[0]  # 能量充足的基站
    B2 = np.where(energy_required >= E)[0]  # 能量不足的基站
    #计算总收集的可再生能源
    excess_energy = E[B1] - energy_required[B1]
    Ec = min(np.sum(alpha_n_a[B1] * excess_energy), max_storage)  # 考虑存储限制
    print(f"总收集的可再生能源: {Ec}")
    # 按损失因子的非升序对 B2 进行排序
    B2_sorted = B2[np.argsort(-alpha_a_n[B2])]
    print(f"需要能量分配的基站 (按优先级排序): {B2_sorted}")
    # 初始化 xi
    xi = E.copy()
    #将能量分配给 B2 中的基站
    for n in B2_sorted:
        required_energy = energy_required[n] - E[n]
        print(f"基站 {n} 需要的能量: {required_energy}")
        if alpha_a_n[n] * Ec >= required_energy:
            #如果聚合器有足够的能量，则分配所需能量
            transfer_energy = required_energy / alpha_a_n[n]
            xi[n] += required_energy
            Ec -= transfer_energy
            print(f"向基站 {n} 分配能量: {required_energy}，剩余聚合器能量: {Ec}")
        else:
            #如果聚合器能量不足，则分配剩余能量
            xi[n] += alpha_a_n[n] * Ec
            Ec = 0
            print(f"聚合器能量不足，向基站 {n} 分配剩余能量: {alpha_a_n[n] * Ec}")
            break # 退出循环，因为没有更多能量
    #检查所有 BS 是否满足能量需求
    zeta = 1
    for n in range(len(E)):
        if xi[n] < energy_required[n]:
            zeta = 0
            break
    print(f"能量分配是否成功: {zeta}")
    return xi.tolist(), zeta
def JESLS(BS, Users, R_req, g, P_max, Pc, rho, W, eta, E, alpha_n_a, alpha_a_n, epsilon1=1e-6, max_iter=1000):   
    t = 0
    zeta = 0  # 能量充足
    P = P_max.copy()  # 初始化发射功率为最大值
    PG = sum(max(rho[n] * P[n] + Pc[n] - E[n], 0) for n in range(len(BS)))  # 计算初始电网能量消耗
    # 用于存储每次迭代后的总能量消耗
    PG_history = [PG]
    while t < max_iter:
        t += 1
        P_old = P.copy()  # 保存上一轮的发射功率
        PG_old = PG  # 保存上一轮的电网能量消耗
        # 步骤 3: 计算 R_bar
        R_bar = {}
        for n in range(len(BS)):
            for m in range(len(Users)):
                # 计算干扰项
                interference = sum(P_old[n_prime] * g[n_prime, m] for n_prime in range(len(BS)) if n_prime != n)
                # 计算 R_bar[(n, m)]
                R_bar[(n, m)] = W * np.log2(1 + P_old[n] * g[n, m] / (eta[m] * interference))
        #求解 LDP 
        X, BL, Y = LDP(BS, Users, R_bar, R_req)
        X_array = np.array([[X.get((n, m), 0) for m in Users] for n in BS])
        Y_array = np.array([[Y.get((n, m), 0) for m in Users] for n in BS])
        #求解ECP
        #IPC进行功率控制
        P = IPC(BS, Users, X_array, Y_array, R_req, g, P_max, rho, W, eta)
        #ECP进行能量协作
        xi, zeta = ECP(E, P_old, Pc, rho, alpha_n_a, alpha_a_n)
        #计算新的电网能量消耗
        PG = sum(max(rho[n] * P[n] + Pc[n] - xi[n], 0) for n in range(len(BS)))
        # 记录当前迭代的总能量消耗
        PG_history.append(PG)
        #终止条件
        if zeta == 1 or abs(PG_old - PG) < epsilon1:
            break
    return X, Y, P, xi, PG, t, PG_history

BS_distance = 1000  # 相邻基站之间的距离（米）
P_max = 40  # 基站的最大发射功率（瓦）
num_BS = 2  # 简单两小区网络中的基站数量
# 随机生成基站的能量收集速率
E = np.random.uniform(20,90,num_BS)  # 基站的能量收集速率（瓦）
rho = 2.8571  # 功率放大器的效率
Pc = 10  # 每个基站的静态电路功耗（瓦）
W = 10e6  # 系统的总带宽（赫兹）
alpha_n_a = np.random.uniform(0.7, 0.9, num_BS)  # 基站到聚合器的能量传输效率
alpha_a_n = np.random.uniform(0.7, 0.9, num_BS)  # 聚合器到基站的能量传输效率
eta = 5e-17  # 噪声功率密度（瓦/赫兹）
path_loss_exponent = 4  # 路径损耗指数
shadow_fading_std = 8  # 阴影衰落标准差（分贝）
num_users = 4  # 用户数量示例
R_req = 3e6 * np.ones(num_users)  # 所有用户的速率需求（3 Mb/s）
# 随机生成信道增益
g = np.random.exponential(scale=1.0, size=(num_BS, num_users))
# 应用路径损耗和阴影衰落
distances = np.random.uniform(100, 1000, (num_BS, num_users))  # 基站与用户之间的随机距离
shadow_fading = np.random.normal(0, shadow_fading_std, (num_BS, num_users))  # 阴影衰落
g = g * (distances ** (-path_loss_exponent)) * 10 ** (shadow_fading / 10)  # 计算最终信道增益
# 计算 R_bar
R_bar = {}
for n in range(num_BS):
    for m in range(num_users):
        R_bar[(n, m)] = W * np.log2(1 + P_max * g[n, m] / (eta * W)) # 计算每个用户在每个基站的理论速率
BS = list(range(num_BS))  # 基站列表
Users = list(range(num_users))  # 用户列表
# 将部分参数转换为函数所需的格式
P_max = np.full(num_BS, P_max)  # 每个基站的最大发射功率
Pc = np.full(num_BS, Pc)  # 每个基站的静态电路功耗
rho = np.full(num_BS, rho)  # 每个基站的功率放大器效率
eta = np.full(num_users, eta * W)  # 将噪声功率密度转换为总噪声功率
#JESLS 进行计算
X, Y, P, xi, PG, t, PG_history = JESLS(BS, Users, R_req, g, P_max, Pc, rho, W, eta, E, alpha_n_a, alpha_a_n)
print(f"最终用户关联 (X): {X}")
print(f"资源分配 (Y): {Y}")
print(f"最优发射功率 (P): {P}")
print(f"基站最终能量 (xi): {xi}")
print(f"总电网能量消耗 (PG): {PG}")
print(f"迭代次数: {t}")
# 输出每次迭代后的总能量消耗
print("每次迭代后的总能量消耗 (PG_history):")
for i, pg in enumerate(PG_history):
    print(f"Iteration {i}: {pg}")
# 绘制总能量消耗和迭代次数的关系图
plt.figure(figsize=(8, 5)) 
plt.plot(range(len(PG_history)), PG_history, marker='o', linestyle='-', color='b', label='Total Grid Energy Consumption')
plt.title('Total Grid Energy Consumption vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Total Grid Energy Consumption (W)')
plt.grid(True)  # 显示网格
plt.legend()  # 显示图例
plt.show()