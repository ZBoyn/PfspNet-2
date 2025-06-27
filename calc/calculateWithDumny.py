import numpy as np

# 虚拟交货期
def virtual_due_date(release_times, processing_times, slack_factor=1.5):
    total_processing_times = np.sum(processing_times, axis=1)
    # 得到虚拟交货期
    due_dates = release_times + total_processing_times * slack_factor
    return due_dates

# 问题参数类
class ProblemParameters:
    def __init__(self, M, N, K, P, E, D, R, U, S, W):
        """
        M: 机器数量
        N: 工件数量
        K: 时段数量
        P: 工件加工时间
        E: 工件能量消耗
        D: 工件交货期
        R: 工件释放时间
        U: 时段开始时间
        S: 时段结束时间
        W: 时段电价
        """
        self.M, self.N, self.K = M, N, K
        self.P, self.E, self.D, self.R = np.array(P), np.array(E), np.array(D), np.array(R)
        self.U, self.S, self.W = np.array(U), np.array(S), np.array(W)

class CalculateWithDummy:
    def __init__(self, params: ProblemParameters):
        self.params = params
        
        # 虚拟节点的数量是 K - 1
        self.num_dummy_jobs = params.K - 1
        # 总"物品"（工件+虚拟节点）的数量
        self.num_total_items = params.N + self.num_dummy_jobs

        
        self.P_extended = np.vstack([params.P, np.zeros((self.num_dummy_jobs, params.M))])
        self.E_extended = np.vstack([params.E, np.zeros((self.num_dummy_jobs, params.M))])
        self.R_extended = np.concatenate([params.R, np.zeros(self.num_dummy_jobs)])
        self.D_extended = np.concatenate([params.D, np.zeros(self.num_dummy_jobs)])

        # 预计算每个"物品"在每台机器上的总能耗
        self.energy_consumption = self.P_extended * self.E_extended

    def calculate(self, sequence):
        """
        计算Cmax和TEC, 将不能跨时段限制视作虚拟节点, 加工时间为0, 能量消耗为0
        返回: Cmax, TEC, 总延期
        """
        completion_times = np.zeros((self.num_total_items, self.params.M))
        total_energy_consumption = 0
        total_tardiness = 0

        # 当前所处的时段索引
        current_window_idx = 0
        # 机器可用时间
        machine_availability = np.zeros(self.params.M)

        for item_idx in sequence:
            is_dummy = item_idx >= self.params.N
            print(f"[TRACE] 调度item_idx={item_idx} ({'虚拟节点' if is_dummy else '真实工件'}), current_window_idx={current_window_idx}, machine_availability={machine_availability}, sequence={sequence}")
            if is_dummy:
                # 如果是虚拟节点的话，就跳到对应时段
                # 注意： 这里不一定是下一个时段，因为虚拟节点可能跨越多个时段
                # 譬如说5个真实工件，3个虚拟节点(5, 6, 7)，那么第6个工件是虚拟节点，它跨越了2个时段
                dummy_id = item_idx - self.params.N  # 1
                target_time_idx = dummy_id + 1 # 2
                print(f"[TRACE] 虚拟节点dummy_id={dummy_id}, 跳转到时段target_time_idx={target_time_idx}")
                current_window_idx = max(current_window_idx, target_time_idx) # 跳到了第2个时段

                # 如果没有时段了，就跳出循环
                if current_window_idx >= self.params.K: # 虚拟节点表示的都是时段结束处, current_window_idx >= K 表示没有时段了
                    print(f"[DEBUG] 虚拟节点导致时段越界：current_window_idx={current_window_idx}, K={self.params.K}, item_idx={item_idx}, sequence={sequence}")
                    return float('inf'), float('inf'), float('inf')
            
                continue

            # 处理真实工件
            job_idx = item_idx
            job_completion_on_prev_machine = 0
            for m in range(self.params.M):
                start_time = machine_availability[m]
                if m == 0:
                    start_time = max(start_time, self.R_extended[job_idx])
                else:
                    start_time = max(start_time, job_completion_on_prev_machine)
                start_time = max(start_time, self.params.U[current_window_idx])

                p_jm = self.P_extended[job_idx, m]
                finish_time = start_time + p_jm
                print(f"[TRACE] 工件{job_idx}在机器{m}：start_time={start_time}, p_jm={p_jm}, finish_time={finish_time}, job_completion_on_prev_machine={job_completion_on_prev_machine}, machine_availability={machine_availability}, 时段[{self.params.U[current_window_idx]}, {self.params.S[current_window_idx]}]")
                if finish_time > self.params.S[current_window_idx]:
                    print(f"[DEBUG] 工件{job_idx}在机器{m}的完工时间{finish_time}超过了当前时段{current_window_idx}的结束时间{self.params.S[current_window_idx]}")
                    print(f"    start_time={start_time}, p_jm={p_jm}, machine_availability={machine_availability}, job_completion_on_prev_machine={job_completion_on_prev_machine}, sequence={sequence}")
                    return float('inf'), float('inf'), float('inf')
            
                completion_times[job_idx, m] = finish_time
                job_completion_on_prev_machine = finish_time
                machine_availability[m] = finish_time

                energy = self.energy_consumption[job_idx, m]
                price = self.params.W[current_window_idx]
                total_energy_consumption += energy * price

            final_completion_time = completion_times[job_idx, self.params.M - 1]
            due_date = self.D_extended[job_idx]
            tardiness = max(0, final_completion_time - due_date)
            total_tardiness += tardiness
        
        real_jobs_indices = [idx for idx in sequence if idx < self.params.N]
        if not real_jobs_indices:
            return 0, 0, 0
        
        final_completion_time = completion_times[real_jobs_indices, self.params.M - 1]
        cmax = np.max(final_completion_time)
        
        return cmax, total_energy_consumption, total_tardiness
    
if __name__ == '__main__':
    M, N, K = 3, 4, 3
    P = np.array([[5, 3, 4], [2, 6, 3], [7, 2, 5], [4, 4, 1]])
    E = np.array([[2, 3, 2], [3, 2, 2], [1, 4, 3], [2, 2, 4]])
    R = np.array([0, 15, 2, 8])
    U = np.array([0, 20, 40])
    S = np.array([20, 40, 100])
    W = np.array([1.0, 1.5, 0.8])
    
    print("="*50)
    print(">>> 为预训练准备虚拟交货期...")
    D = virtual_due_date(R, P, slack_factor=1.5)
    print(f"虚拟交货期 D: {D}")
    
    print("="*50)
    print(">>> 创建问题实例和计算器...")
    problem_params = ProblemParameters(M, N, K, P, E, D, R, U, S, W)
    calculator = CalculateWithDummy(problem_params)
    print(f"工件释放时间 R: {R}")
    print("计算器已准备就绪。\n")
    
    print("="*50)
    print(">>> 测试一个调度序列...")
    test_sequence = [2, 4, 0, 1, 3, 5]
    cmax, tec, tardiness = calculator.calculate(test_sequence)
    
    print(f"测试序列: {test_sequence}")
    print("-"*40)
    if cmax == float('inf'):
        print("该序列是无效的。")
    else:
        print(f"最大完工时间 (Cmax): {cmax:.2f}")
        print(f"总能耗成本 (TEC): {tec:.2f}")
        print(f"总延期时间 (Total Tardiness): {tardiness:.2f}")
    print("="*50)
