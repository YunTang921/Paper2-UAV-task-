# 作用: 使用龙蜻+差分进化算法离线求多无人机任务分配; 依赖: 无其它项目文件; 被依赖: 无
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

class DragonflyDEAlgorithm:
    def __init__(self, num_drones, num_targets, max_iter, population_size, tasks_data, w1=0.5, w2=0.5):
        self.num_drones = num_drones
        self.num_targets = num_targets
        self.num_tasks = num_targets * 3  # 每个目标包含三个任务（S、D、I）
        self.max_iter = max_iter
        self.population_size = population_size
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.fitness_history = []
        self.tasks_data = tasks_data  # 从文件中加载的任务点数据
        self.depot = np.array([0, 0])  # 仓库坐标
        self.w1 = w1  # 每架无人机的航迹距离权重
        self.w2 = w2  # 无人机之间的航迹距离差异权重

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = np.zeros(self.num_tasks)
            for i in range(self.num_tasks):
                solution[i] = i % self.num_drones + np.random.rand()  # 确保初始任务均匀分配
            np.random.shuffle(solution)  # 打乱初始解顺序
            population.append(solution)
        return np.array(population)

    def decode_solution(self, solution):
        tasks_allocation = [[] for _ in range(self.num_drones)]
        task_sequences = [[] for _ in range(self.num_drones)]
        for task_idx in range(self.num_tasks):
            drone_id = int(solution[task_idx])
            sequence_value = solution[task_idx] - drone_id
            tasks_allocation[drone_id].append(task_idx)
            task_sequences[drone_id].append((task_idx, sequence_value))
        for drone_id in range(self.num_drones):
            # 根据小数部分排序
            task_sequences[drone_id].sort(key=lambda x: x[1])
        return task_sequences

    def evaluate_fitness(self, solution):
        total_distance = 0
        distances = []
        task_sequences = self.decode_solution(solution)
        for drone_id in range(self.num_drones):
            route_distance = 0
            tasks = [task for task, _ in task_sequences[drone_id]]
            if len(tasks) > 0:
                # 计算从仓库到第一个任务点的距离
                route_distance += np.linalg.norm(self.depot - self.tasks_data[tasks[0]][:2])
                # 计算任务点之间的距离
                for i in range(len(tasks) - 1):
                    route_distance += np.linalg.norm(
                        self.tasks_data[tasks[i]][:2] - self.tasks_data[tasks[i + 1]][:2])
                # 计算从最后一个任务点返回仓库的距离
                route_distance += np.linalg.norm(self.tasks_data[tasks[-1]][:2] - self.depot)
            total_distance += route_distance
            distances.append(route_distance)
        # 计算无人机之间航迹距离的差异
        distance_variance = np.var(distances)
        # 计算加权适应度
        fitness = self.w1 * total_distance + self.w2 * distance_variance
        return fitness, total_distance, distances

    def levy_flight(self, beta):
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.num_tasks)
        v = np.random.normal(0, 1, size=self.num_tasks)
        step = u / abs(v) ** (1 / beta)
        return step

    def update_position(self, individual, global_best, neighbor_best, beta=1.5):
        s = np.zeros_like(individual)  # Separation
        a = np.zeros_like(individual)  # Alignment
        c = np.zeros_like(individual)  # Cohesion
        f = np.zeros_like(individual)  # Attraction
        e = np.zeros_like(individual)  # Enemy avoidance
        s = individual - neighbor_best
        a = (global_best + neighbor_best) / 2 - individual
        c = (global_best + neighbor_best) / 2 - individual
        f = global_best - individual
        e = neighbor_best - individual
        position_update = individual + s + a + c + f + e + self.levy_flight(beta)
        position_update = np.clip(position_update, 0, self.num_drones - 1 + 0.999999)  # 确保索引在有效范围内
        return position_update

    def differential_evolution_operator(self, population, global_best):
        new_population = []
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = population[a] + self.F * (population[b] - population[c])
            mutant = np.clip(mutant, 0, self.num_drones - 1 + 0.999999)  # 保持变异后的值在任务索引范围内
            trial = np.copy(population[i])
            for j in range(len(trial)):
                if np.random.rand() < self.CR:
                    trial[j] = mutant[j]
            # 使用蜻蜓算法更新位置
            trial = self.update_position(trial, global_best, population[i])
            # 检查并确保任务分配均衡
            task_sequences = self.decode_solution(trial)
            tasks_per_drone = [len(tasks) for tasks in task_sequences]
            if max(tasks_per_drone) - min(tasks_per_drone) > 5:  # 如果任务分配差异太大，重新生成
                trial = np.random.rand(self.num_tasks) * self.num_drones
            new_population.append(trial)
        return np.array(new_population)

    def optimize(self):
        global_best = self.population[0]
        global_best_fitness, _, _ = self.evaluate_fitness(global_best)
        for iter in range(self.max_iter):
            print(f"Iteration {iter + 1}/{self.max_iter}")
            new_population = self.differential_evolution_operator(self.population, global_best)
            for i in range(self.population_size):
                trial_fitness, _, _ = self.evaluate_fitness(new_population[i])
                current_fitness, _, _ = self.evaluate_fitness(self.population[i])
                if trial_fitness < current_fitness:
                    self.population[i] = new_population[i]
                    if trial_fitness < self.best_fitness:
                        self.best_solution = new_population[i]
                        self.best_fitness = trial_fitness
                        global_best = new_population[i]
                        global_best_fitness = trial_fitness
            print(f"Best Fitness after iteration {iter + 1}: {self.best_fitness}")
            self.fitness_history.append(self.best_fitness)
        # 确保 self.best_solution 被更新
        if self.best_solution is None:
            self.best_solution = global_best
            self.best_fitness = global_best_fitness
        return self.best_solution, self.best_fitness

    def print_solution(self):
        task_sequences = self.decode_solution(self.best_solution)
        _, total_distance, distances = self.evaluate_fitness(self.best_solution)
        for drone_id in range(self.num_drones):
            tasks = task_sequences[drone_id]
            tasks = [self.get_task_type(task) for task, _ in tasks]
            print(f"无人机 {drone_id + 1} 的任务执行方案: {tasks}")
            print(f"无人机 {drone_id + 1} 的航程: {distances[drone_id]:.2f}")
        print(f"所有无人机的总航程: {total_distance:.2f}")

    def get_task_type(self, task_idx):
        target_idx = task_idx // 3
        task_type_idx = task_idx % 3
        task_type = ['S', 'D', 'I'][task_type_idx]
        return f"目标 {target_idx + 1} 的 {task_type} 任务"

    def plot_fitness(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, label='Best Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Convergence Curve')
        plt.legend()
        plt.grid()
        plt.show()

    def get_initial_solution_tensor(self):
        task_sequences = self.decode_solution(self.best_solution)
        max_tasks = max(len(tasks) for tasks in task_sequences)
        initial_solutions = np.zeros((self.num_drones, max_tasks, 8))
        for drone_id in range(self.num_drones):
            tasks = [task for task, _ in task_sequences[drone_id]]
            for i, task in enumerate(tasks):
                task_type_encoding = [0, 0, 0]
                if task % 3 == 0:
                    task_type_encoding = [0, 0, 1]  # S任务
                elif task % 3 == 1:
                    task_type_encoding = [0, 1, 0]  # D任务
                elif task % 3 == 2:
                    task_type_encoding = [1, 0, 0]  # I任务
                initial_solutions[drone_id, i, :2] = self.tasks_data[task][:2]  # 初始位置坐标
                initial_solutions[drone_id, i, 2] = self.tasks_data[task][2]  # 任务半径
                initial_solutions[drone_id, i, 3:5] = self.tasks_data[task][:2]  # 结束位置坐标（假设相同）
                initial_solutions[drone_id, i, 5:8] = task_type_encoding  # 任务类型编码
        return torch.tensor(initial_solutions, dtype=torch.float32)

# 从文件中读取任务坐标信息
def load_task_coordinates(file_path):
    tasks_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Data: tensor'):
                # 提取数据的后三行
                for j in range(1, 4):
                    line = lines[i + j].strip().replace('tensor([', '').replace('])', '').replace('[', '').replace(']', '')
                    values = [value.strip() for value in line.split(',') if value.strip()]
                    if len(values) >= 3:
                        x, y, r = float(values[0]), float(values[1]), float(values[2])
                        tasks_data.append([x, y, r])
                    else:
                        print(f"Warning: Skipping line due to insufficient data: {line}")
    return np.array(tasks_data)

# 参数设置
num_drones = 3
num_targets = 12
max_iter = 100
population_size = 100

# 权重设置（确保它们的和为1）
w1 = 0.6  # 最小化总航迹距离的权重
w2 = 0.4  # 最小化无人机之间的航迹差异的权重

# 读取保存的任务数据
tasks_data = load_task_coordinates('task_info.txt')

# 初始化并运行算法
da_de = DragonflyDEAlgorithm(num_drones, num_targets, max_iter, population_size, tasks_data, w1, w2)
best_solution, best_fitness = da_de.optimize()

print("最优任务分配方案:")
da_de.print_solution()

# 获取初始任务分配方案的张量格式
initial_solutions_tensor = da_de.get_initial_solution_tensor()
print("初始任务分配方案张量格式:")
print(initial_solutions_tensor)

# 绘制收敛图
da_de.plot_fitness()
