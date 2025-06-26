import numpy as np

class ParetoArchive:
    def __init__(self, capacity):
        self.capacity = capacity  # 存档的最大容量
        self.solutions = []       # 存储解的 [Cmax, TEC]
        self.sequences = []       # 存储对应的工件序列

    def add(self, new_solution, new_sequence):
        # new_solution 是一个类似 [cmax, tec] 的列表或元组
        is_dominated_by_archive = False
        to_be_removed_indices = []

        # 1. 检查新解是否被存档中的解支配
        for i, sol in enumerate(self.solutions):
            if np.all(np.array(sol) <= np.array(new_solution)) and np.any(np.array(sol) < np.array(new_solution)):
                is_dominated_by_archive = True
                break
        
        if is_dominated_by_archive:
            return False # 新解被支配，不添加

        # 2. 检查新解支配了存档中的哪些解
        for i, sol in enumerate(self.solutions):
            if np.all(np.array(new_solution) <= np.array(sol)) and np.any(np.array(new_solution) < np.array(sol)):
                to_be_removed_indices.append(i)
        
        # 3. 移除被新解支配的旧解
        # 从后往前删除，避免索引错乱
        for i in sorted(to_be_removed_indices, reverse=True):
            del self.solutions[i]
            del self.sequences[i]
            
        # 4. 添加新解
        self.solutions.append(new_solution)
        self.sequences.append(new_sequence)
        
        # 5. 如果存档超出容量，进行修剪 (Pruning)
        if len(self.solutions) > self.capacity:
            self.prune()
            
        return True

    def prune(self):
        # 修剪策略：移除最拥挤的解，以保持多样性
        # 这里使用一种简化的“拥挤度计算”方法
        if len(self.solutions) <= 2:
            return

        # 按第一个目标（Cmax）排序
        sorted_indices = sorted(range(len(self.solutions)), key=lambda k: self.solutions[k][0])
        
        distances = [float('inf')] * len(self.solutions)
        
        cmax_min = self.solutions[sorted_indices[0]][0]
        cmax_max = self.solutions[sorted_indices[-1]][0]
        tec_min = min(s[1] for s in self.solutions)
        tec_max = max(s[1] for s in self.solutions)

        for i in range(1, len(sorted_indices) - 1):
            dist = (self.solutions[sorted_indices[i+1]][0] - self.solutions[sorted_indices[i-1]][0]) / (cmax_max - cmax_min + 1e-9)
            dist += (self.solutions[sorted_indices[i-1]][1] - self.solutions[sorted_indices[i+1]][1]) / (tec_max - tec_min + 1e-9) # TEC是反向的
            distances[sorted_indices[i]] = dist
            
        # 找到距离最小（最拥挤）的解并移除
        crowded_index = np.argmin(distances)
        del self.solutions[crowded_index]
        del self.sequences[crowded_index]

    def get_front(self):
        return np.array(self.solutions)