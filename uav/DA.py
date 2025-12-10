import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义三层BP神经网络
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化蜻蜓群体
def initialize_dragonflies(N, D):
    P = [np.random.rand(D) for _ in range(N)]
    V = [np.zeros(D) for _ in range(N)]
    return P, V


# 计算适应度
def evaluate_fitness(P, model, loss_fn, X, y):
    param_count = 0
    for param in model.parameters():
        param_numel = param.data.numel()
        param.data = torch.tensor(P[param_count:param_count + param_numel], dtype=torch.float32).view_as(param.data)
        param_count += param_numel
    model = model.cuda()
    rewards, log_probs, actions = model(X)
    loss = loss_fn(rewards, y)
    return loss.item()


# 更新蜻蜓群体
def update_dragonflies(P, V, best_P, worst_P, w, s, a, c, f, e):
    N = len(P)
    D = len(P[0])
    new_P = np.copy(P)
    new_V = np.copy(V)

    for i in range(N):
        S = -np.sum([P[i] - P[j] for j in range(N) if j != i], axis=0)
        A = np.mean([V[j] for j in range(N) if j != i], axis=0)
        C = np.mean([P[j] for j in range(N) if j != i], axis=0) - P[i]
        F = best_P - P[i]
        E = P[i] - worst_P

        new_V[i] = w * V[i] + s * S + a * A + c * C + f * F + e * E
        new_P[i] = P[i] + new_V[i]

    return new_P, new_V


# 主优化过程（蜻蜓算法）
def dragonfly_algorithm(model, X, y, N, D, max_iterations, w, s, a, c, f, e):
    P, V = initialize_dragonflies(N, D)
    best_P = None
    best_fitness = float('inf')
    worst_P = None
    worst_fitness = float('-inf')
    loss_fn = nn.MSELoss()

    for iteration in range(max_iterations):
        print('迭代次数:',iteration)
        fitnesses = []
        for i in range(N):
            fitness = evaluate_fitness(P[i], model, loss_fn, X, y)
            fitnesses.append(fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_P = P[i]
            if fitness > worst_fitness:
                worst_fitness = fitness
                worst_P = P[i]

        P, V = update_dragonflies(P, V, best_P, worst_P, w, s, a, c, f, e)

    return best_P


# 梯度下降优化
def gradient_descent_optimization(model, X, y, lr=0.01, epochs=100):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
