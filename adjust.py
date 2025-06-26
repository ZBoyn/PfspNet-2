import numpy as np
import itertools

# 使用一个简单的类来存储每个解，比字典更清晰
class Solution:
    """封装一个调度解"""
    def __init__(self, sequence):
        self.sequence = np.array(sequence)
        self.C = None         # 完工时间矩阵 (M x N)
        self.period = None    # 周期矩阵 (M x N)
        self.Y = None         # 机器在各周期是否工作的标志 (K x M)
        self.TEC = 0.0        # 总能耗
        self.TCA = 0.0        # 总成本

    def __repr__(self):
        return f"Solution(sequence={self.sequence}, TCA={self.TCA:.2f})"

class ScheduleOptimizer:
    """
    一个用于处理考虑分时电价的置换流水车间调度问题的优化器。
    该类实现了从 C++ 代码转换而来的 `adjust_initial` 逻辑。
    """
    def __init__(self, M, N, K, P, E, R, U, S, W):
        """
        初始化优化器所需的所有参数。

        :param M: 机器数量
        :param N: 工件数量
        :param K: 电价周期数量
        :param P: 加工时间矩阵 (M x N), P[i, j] 是工件 j 在机器 i 上的加工时间
        :param E: 加工功率矩阵 (M x N), E[i, j] 是工件 j 在机器 i 上的功率
        :param R: 工件释放时间数组 (N), R[j] 是工件 j 的最早可用时间
        :param U: 电价周期结束时间点数组 (K+1)
        :param S: 电价周期开始时间点数组 (K+1)
        :param W: 各周期电价数组 (K)
        """
        self.M, self.N, self.K = M, N, K
        self.P, self.E, self.R = np.array(P), np.array(E), np.array(R)
        self.U, self.S, self.W = np.array(U), np.array(S), np.array(W)
        
        # 预先计算每个工件在每台机器上的能耗
        self.energy_consumption = self.P * self.E

    def adjust_initial_schedule(self, population):
        """
        对种群中的每一个解（工件序列）进行时间和成本优化。
        这是 C++ `adjustInitial` 函数的主要逻辑转换。
        """
        for p, solution in enumerate(population):
            print(f"\n--- 正在优化解 {p+1}: 序列 {solution.sequence} ---")
            
            # 初始化存储矩阵
            last_time = np.zeros((self.M, self.N), dtype=int)
            last_period = np.zeros((self.M, self.N), dtype=int)
            first_time = np.zeros((self.M, self.N), dtype=int)
            first_period = np.zeros((self.M, self.N), dtype=int)
            
            # C++中的pop[p].sequence[j] - 1 在这里就是 solution.sequence[j]
            # 因为Python的列表/Numpy数组本身就是0-indexed
            
            # =================================================================
            # 1. 计算最晚时间 (Backward Pass ⏪)
            # =================================================================
            for i in range(self.M - 1, -1, -1):
                for j in range(self.N - 1, -1, -1):
                    job_id = solution.sequence[j]
                    
                    if i == self.M - 1: # === 最后一行机器 ===
                        if j == self.N - 1: # 最后一个工序
                            last_time[i, j] = self.U[self.K] - self.S[self.K - 1]
                            last_period[i, j] = self.K
                        else: # 其他工序
                            next_job_id = solution.sequence[j + 1]
                            last_time[i, j] = last_time[i, j + 1] - self.P[i, next_job_id]
                            last_period[i, j] = last_period[i, j + 1]
                        
                        # 检查周期容量是否足够
                        while last_time[i, j] - self.U[last_period[i, j] - 1] < self.P[i, job_id]:
                            last_period[i, j] -= 1
                            last_time[i, j] = self.U[last_period[i, j]] - self.S[last_period[i, j] - 1]
                    
                    else: # === 非最后一行机器 ===
                        # 约束来自下一台机器上的同一个工件
                        last_time[i, j] = last_time[i + 1, j] - self.P[i + 1, job_id]
                        last_period[i, j] = last_period[i + 1, j]
                        
                        # 检查周期容量
                        while last_time[i, j] - self.U[last_period[i, j] - 1] < self.P[i, job_id]:
                            last_period[i, j] -= 1
                            last_time[i, j] = self.U[last_period[i, j]] - self.S[last_period[i, j] - 1]
                        
                        if j != self.N - 1: # 同时受同一台机器上后一个工件的约束
                            next_job_id = solution.sequence[j + 1]
                            next_job_start_time = last_time[i, j + 1] - self.P[i, next_job_id]
                            
                            if next_job_start_time < last_time[i, j]:
                                last_time[i, j] = next_job_start_time
                                last_period[i, j] = last_period[i, j + 1]
                                
                                # 再次检查周期容量
                                while last_time[i, j] - self.U[last_period[i, j] - 1] < self.P[i, job_id]:
                                    last_period[i, j] -= 1
                                    last_time[i, j] = self.U[last_period[i, j]] - self.S[last_period[i, j] - 1]

            # =================================================================
            # 2. 计算最早时间 (Forward Pass ⏩)
            # =================================================================
            for i in range(self.M):
                for j in range(self.N):
                    job_id = solution.sequence[j]
                    
                    # 计算基础最早开始时间
                    if i == 0 and j == 0:
                        start_time = self.R[job_id]
                    elif i == 0:
                        prev_job_id = solution.sequence[j - 1]
                        start_time = max(first_time[i, j - 1] + self.P[i, prev_job_id], self.R[job_id])
                    elif j == 0:
                        start_time = first_time[i - 1, j] + self.P[i - 1, job_id]
                    else:
                        prev_job_id_on_same_machine = solution.sequence[j - 1]
                        finish_time_prev_job = first_time[i, j - 1] + self.P[i, prev_job_id_on_same_machine]
                        finish_time_on_prev_machine = first_time[i - 1, j] + self.P[i - 1, job_id]
                        start_time = max(finish_time_prev_job, finish_time_on_prev_machine)
                    
                    first_time[i,j] = start_time
                    
                    # 确定最早周期，处理跨周期情况
                    for k in range(1, self.K + 1):
                        if self.U[k] > start_time:
                            if self.U[k] - start_time >= self.P[i, job_id]:
                                first_period[i, j] = k
                            else:
                                first_time[i,j] = self.U[k] # 推迟到下个周期开始
                                first_period[i, j] = k + 1
                            break

            # =================================================================
            # 3. 调度策略 (Decision Making ⚙️)
            # =================================================================
            solution.C = np.zeros((self.M, self.N), dtype=int)
            solution.period = np.zeros((self.M, self.N), dtype=int)
            
            for i in range(self.M - 1, -1, -1):
                for j in range(self.N - 1, -1, -1):
                    job_id = solution.sequence[j]
                    
                    # === 第一步: 确定决策空间 (可行的电价周期) ===
                    start_p = first_period[i, j] -1
                    
                    end_p = last_period[i, j] - 1
                    if j < self.N - 1:
                        next_job_id = solution.sequence[j + 1]
                        end_p = min(end_p, solution.period[i, next_job_id] -1)
                    if i < self.M - 1:
                        end_p = min(end_p, solution.period[i + 1, job_id] -1)
                    
                    # 确保 start_p 不大于 end_p
                    if start_p > end_p:
                        # 这种情况理论上不应发生，如果发生了说明存在问题
                        # 默认使用最早周期作为唯一选择
                        end_p = start_p
                        
                    feasible_periods = list(range(int(start_p), int(end_p) + 1))

                    # === 第二步: 优化搜索顺序 (按成本排序) ===
                    feasible_periods.sort(key=lambda p: (self.W[p], p))

                    # === 第三步: 试探-验证决策 ===
                    best_period_found = False
                    for candidate_period_idx in feasible_periods:
                        candidate_period = candidate_period_idx + 1 # 周期是1-based

                        # 计算在此约束下的最晚完工时间 C
                        # 约束来自后继者
                        c_max = float('inf')
                        if j < self.N - 1:
                            next_job_id = solution.sequence[j + 1]
                            c_max = min(c_max, solution.C[i, next_job_id] - self.P[i, next_job_id])
                        if i < self.M - 1:
                            c_max = min(c_max, solution.C[i + 1, job_id] - self.P[i + 1, job_id])
                        if c_max == float('inf'): # 最后一个工序
                             c_max = last_time[i, j]
                        
                        # 核心验证
                        completion_time = c_max
                        
                        # 检查特殊优化：如果选的周期比后继者早，可以排满整个周期
                        is_earlier_period = True
                        if j < self.N - 1:
                            next_job_id = solution.sequence[j+1]
                            if candidate_period == solution.period[i, next_job_id]:
                                is_earlier_period = False
                        if i < self.M - 1:
                            if candidate_period == solution.period[i+1, job_id]:
                                is_earlier_period = False
                        
                        if is_earlier_period:
                            # 拥有整个周期的使用权，将完工时间设为周期末尾
                            completion_time = self.U[candidate_period]
                        
                        # 验证时间容量
                        if completion_time - self.U[candidate_period - 1] >= self.P[i, job_id]:
                            solution.C[i, job_id] = completion_time
                            solution.period[i, job_id] = candidate_period
                            best_period_found = True
                            break # 找到最便宜且可行的，跳出循环
                    
                    if not best_period_found:
                        print(f"警告: 未能为工序 (机器{i}, 工件{job_id}) 找到可行调度！")


            # =================================================================
            # 4. Finalization and Evaluation
            # =================================================================
            solution.Y = np.zeros((self.K, self.M), dtype=int)
            for m in range(self.M):
                # 获取这台机器用到的所有周期
                used_periods = np.unique(solution.period[m, :])
                for p_id in used_periods:
                    if p_id > 0: # 周期是1-based
                        solution.Y[p_id - 1, m] = 1

            self._calculate_tec(solution)
            self._calculate_tca(solution)
            
        return population

    def _calculate_tec(self, solution):
        """计算总能耗 (Total Energy Consumption)"""
        total_energy = 0
        for i in range(self.M):
            for j in range(self.N):
                job_id = solution.sequence[j]
                total_energy += self.energy_consumption[i, job_id]
        solution.TEC = total_energy
        return total_energy

    def _calculate_tca(self, solution):
        """计算总成本 (Total Cost of Appliances/Activity)"""
        total_cost = 0
        for k in range(self.K):
            for i in range(self.M):
                if solution.Y[k, i] == 1: # 如果机器 i 在周期 k 工作
                    # 成本 = 该周期内所有工件的能耗 * 该周期电价
                    energy_in_period = 0
                    for j_idx in range(self.N):
                        job_id_in_seq = solution.sequence[j_idx]
                        if solution.period[i, job_id_in_seq] == (k + 1):
                           energy_in_period += self.energy_consumption[i, job_id_in_seq]
                    total_cost += energy_in_period * self.W[k]
        solution.TCA = total_cost
        return total_cost


# =================================================================
# 测试算例
# =================================================================
if __name__ == '__main__':
    M = 3  # 3 台机器
    N = 4  # 4 个工件
    K = 5  # 5 个电价周期

    # 加工时间 P (M x N)
    P = [[5, 4, 8, 3], 
         [6, 3, 7, 5], 
         [4, 6, 5, 7]]
    
    # 加工功率 E (M x N)
    E = [[3, 5, 2, 6], 
         [4, 6, 3, 2], 
         [5, 4, 6, 3]]
    
    # 工件释放时间 R (N)
    R = [0, 2, 5, 4]

    # 电价周期定义 (U, S, W)
    # 周期1: 0-10, 周期2: 10-25, 周期3: 25-40, 周期4: 40-55, 周期5: 55-70
    S = [0, 0, 10, 25, 40, 55] # S[k] 是周期k的开始时间
    U = [0, 10, 25, 40, 55, 70] # U[k] 是周期k的结束时间
    W = [3.0, 5.0, 2.0, 5.0, 3.0] # 周期3最便宜，2和4最贵

    # 2. 创建初始种群 (两个解)
    # 注意工件ID从0开始 (0, 1, 2, 3)
    initial_sequences = [
        [0, 1, 2, 3],
        [3, 1, 0, 2]
    ]
    population = [Solution(seq) for seq in initial_sequences]

    print("="*20 + " 初始化优化器 " + "="*20)
    optimizer = ScheduleOptimizer(M, N, K, P, E, R, U, S, W)
    
    print("\n" + "="*20 + " 开始优化调度 " + "="*20)
    optimized_population = optimizer.adjust_initial_schedule(population)

    print("\n" + "="*20 + " 优化结果展示 " + "="*20)
    for i, sol in enumerate(optimized_population):
        print(f"\n--- 最终解 {i+1} ---")
        print(f"工件序列: {sol.sequence}")
        print(f"总能耗 (TEC): {sol.TEC:.2f}")
        print(f"总成本 (TCA): {sol.TCA:.2f}")
        print("最终完工时间矩阵 C (行:机器, 列:工件ID):")
        # 重新整理矩阵，使其列索引对应工件ID
        C_sorted = np.zeros((M,N))
        for m_idx in range(M):
            for j_idx in range(N):
                C_sorted[m_idx, j_idx] = sol.C[m_idx, j_idx]
        print(np.round(C_sorted).astype(int))
        
        print("最终所在周期矩阵 period (行:机器, 列:工件ID):")
        period_sorted = np.zeros((M,N))
        for m_idx in range(M):
            for j_idx in range(N):
                period_sorted[m_idx, j_idx] = sol.period[m_idx, j_idx]
        print(np.round(period_sorted).astype(int))