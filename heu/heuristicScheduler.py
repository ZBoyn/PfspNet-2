import numpy as np

class HeuristicScheduler:
    """
    一个高质量的启发式调度器，用于在DRL评估阶段解码工件序列。
    它为给定的序列寻找近似最优的 Cmax 和 TEC。
    """
    def __init__(self, M, N, K, P, E, R, U, S, W):
        self.M, self.N, self.K = M, N, K
        self.P, self.E, self.R = np.array(P), np.array(E), np.array(R)
        self.U, self.S, self.W = np.array(U), np.array(S), np.array(W)
        self.energy_consumption = self.P * self.E

    def evaluate(self, sequences):
        """
        评估一批工件序列，返回每个序列的 Cmax 和 TEC。

        :param sequences: 一个包含多个工件序列的列表，例如 [[0,1,2], [2,1,0]]
        :return: (cmax_list, tec_list)
        """
        all_cmax = []
        all_tec = []
        all_c = []

        for seq in sequences:
            solution = self._schedule_one_sequence(np.array(seq))
            cmax, tec, c = self._calculate_objectives(solution)
            all_cmax.append(cmax)
            all_tec.append(tec)
            all_c.append(c)
            
        return all_cmax, all_tec, all_c

    def _schedule_one_sequence(self, sequence):
        """对单个序列进行完整的启发式调度。"""
        # 数据结构初始化
        last_time = np.zeros((self.M, self.N), dtype=int)
        last_period = np.zeros((self.M, self.N), dtype=int)
        first_time = np.zeros((self.M, self.N), dtype=int)
        first_period = np.zeros((self.M, self.N), dtype=int)
        
        # 1. 后向传递，计算最晚时间
        for i in range(self.M - 1, -1, -1):
            for j in range(self.N - 1, -1, -1):
                job_id = sequence[j]
                if i == self.M - 1:
                    if j == self.N - 1:
                        last_time[i, j] = self.U[self.K]
                        last_period[i, j] = self.K
                    else:
                        next_job_id = sequence[j + 1]
                        last_time[i, j] = last_time[i, j + 1] - self.P[i, next_job_id]
                        last_period[i, j] = last_period[i, j + 1]
                    while last_time[i, j] - self.U[last_period[i, j] - 1] < self.P[i, job_id]:
                        last_period[i, j] -= 1
                        last_time[i, j] = self.U[last_period[i, j]]
                else:
                    last_time[i, j] = last_time[i + 1, j] - self.P[i + 1, job_id]
                    last_period[i, j] = last_period[i + 1, j]
                    while last_time[i, j] - self.U[last_period[i, j] - 1] < self.P[i, job_id]:
                        last_period[i, j] -= 1
                        last_time[i, j] = self.U[last_period[i, j]]
                    if j < self.N - 1:
                        next_job_id = sequence[j + 1]
                        next_job_start_time = last_time[i, j + 1] - self.P[i, next_job_id]
                        if next_job_start_time < last_time[i, j]:
                            last_time[i, j] = next_job_start_time
                            last_period[i, j] = last_period[i, j + 1]
                            while last_time[i, j] - self.U[last_period[i, j] - 1] < self.P[i, job_id]:
                                last_period[i, j] -= 1
                                last_time[i, j] = self.U[last_period[i, j]]

        # 2. 正向传递，计算最早时间
        for i in range(self.M):
            for j in range(self.N):
                job_id = sequence[j]
                if i == 0 and j == 0: start_time = self.R[job_id]
                elif i == 0: start_time = first_time[i, j - 1] + self.P[i, sequence[j - 1]]
                elif j == 0: start_time = first_time[i - 1, j] + self.P[i - 1, sequence[j]]
                else:
                    start_time = max(first_time[i, j - 1] + self.P[i, sequence[j-1]], first_time[i - 1, j] + self.P[i - 1, job_id])
                
                start_time = max(start_time, self.R[job_id])
                first_time[i, j] = start_time
                for k in range(1, self.K + 1):
                    if self.U[k] >= start_time:
                        if self.U[k] - start_time >= self.P[i, job_id]:
                            first_period[i, j] = k; break
                        else:
                            first_time[i, j] = self.U[k]; first_period[i, j] = k + 1; break
        
        # 3. 策略调度
        C = np.zeros((self.M, self.N), dtype=int)
        period = np.zeros((self.M, self.N), dtype=int)
        for i in range(self.M - 1, -1, -1):
            for j in range(self.N - 1, -1, -1):
                job_id = sequence[j]
                start_p = int(first_period[i, j] - 1)
                end_p = int(last_period[i, j] - 1)
                if j < self.N - 1: end_p = min(end_p, int(period[i, sequence[j+1]]-1))
                if i < self.M - 1: end_p = min(end_p, int(period[i+1, job_id]-1))

                feasible_periods = list(range(start_p, end_p + 1))
                feasible_periods.sort(key=lambda p_idx: (self.W[p_idx], p_idx))

                for p_idx in feasible_periods:
                    candidate_period = p_idx + 1
                    c_max = float('inf')
                    if j < self.N - 1: c_max = min(c_max, C[i, sequence[j+1]] - self.P[i, sequence[j+1]])
                    if i < self.M - 1: c_max = min(c_max, C[i+1, job_id] - self.P[i+1, job_id])
                    if c_max == float('inf'): c_max = last_time[i, j]
                    
                    completion_time = c_max
                    is_earlier = True
                    if j < self.N - 1 and candidate_period == period[i, sequence[j+1]]: is_earlier = False
                    if i < self.M - 1 and candidate_period == period[i+1, job_id]: is_earlier = False

                    if is_earlier: completion_time = self.U[candidate_period]

                    if completion_time - self.U[candidate_period-1] >= self.P[i, job_id]:
                        C[i, job_id] = completion_time
                        period[i, job_id] = candidate_period
                        break
        return {"C": C, "period": period}

    def _calculate_objectives(self, solution):
        C, period = solution["C"], solution["period"]
        cmax = C[:, -1].max()
        
        tec = 0.0
        for i in range(self.M):
            for j in range(self.N):
                p = int(period[i, j])
                if p > 0:
                    tec += self.energy_consumption[i, j] * self.W[p - 1]
        
        # 如果调度失败，返回无穷大
        if cmax == 0: return float('inf'), float('inf')
        return cmax, tec, C

if __name__ == "__main__":
    # 1. Basic Parameters
    n_jobs = 7
    m_machines = 3
    k_intervals = 5
    # The job sequence is 1-indexed, convert to 0-indexed for Python
    # Original JS = [3, 1, 4, 5, 6, 7, 2]
    JS_0_indexed = [2, 0, 3, 4, 5, 6, 1] 

    # 2. Time and Energy Parameters
    # U needs K+1 elements, where U[k] is the END time of period k.
    # U[0] is the start (typically 0). We add a final horizon time.
    u_starts = np.array([0, 30, 60, 80, 100]) 
    final_horizon = 200 # A reasonable horizon beyond which we don't schedule
    U = np.concatenate((u_starts, [final_horizon]))

    # S (period_duration) is unused in the provided code but required for init
    S = np.array([30, 30, 20, 20, 20]) 
    # W is the energy cost multiplier for each of the K periods
    W = np.array([2, 5, 3, 4, 1]) 

    # 3. Job-specific Parameters
    R = np.array([14, 85, 0, 16, 20, 40, 50]) # Release Times

    # P and E need to be in shape (M, N) = (machines, jobs)
    # The input is (jobs, machines), so we need to transpose them.
    base_processing_time_N_M = np.array([
        [4, 6, 4], [6, 1, 1], [5, 3, 2], [2, 5, 2],
        [5, 2, 3], [6, 7, 5], [9, 10, 4]
    ])
    base_energy_N_M = np.array([
        [7, 4, 4], [3, 3, 3], [4, 3, 6], [2, 6, 3],
        [7, 2, 6], [2, 5, 3], [8, 2, 5]
    ])

    P = base_processing_time_N_M.T
    E = base_energy_N_M.T

    # --- Execution ---

    # Instantiate the scheduler with the formatted data
    scheduler = HeuristicScheduler(
        M=m_machines, N=n_jobs, K=k_intervals,
        P=P, E=E, R=R, U=U, S=S, W=W
    )

    # The evaluate method expects a list of sequences
    sequences_to_run = [JS_0_indexed]

    # Run the evaluation
    cmax_results, tec_results, c = scheduler.evaluate(sequences_to_run)

    # --- Output ---
    print(f"Job Sequence (0-indexed): {sequences_to_run[0]}")
    print("-" * 30)
    print(f"Calculated Cmax: {cmax_results[0]}")
    print(f"Calculated TEC: {tec_results[0]}")
    print(f"Completion Times (C):\n{c[0]}")
    print(f"Periods:\n{c[0]['period']}")
    print("-" * 30)