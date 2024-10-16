import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils import save_convergence_history, save_performance_metrics
from optimization_utils import evaluate_parallel, evaluate_with_cache
from optimization_utils import apply_adaptive_sa




# PSO 的粒子类定义
class Particle:
    def __init__(self, num_positions, num_plates):
        self.position = np.random.randint(0, num_positions, size=num_plates)  # 随机初始化位置
        self.velocity = np.zeros(num_plates)
        self.best_position = self.position.copy()
        self.best_score = np.inf

    def update_velocity(self, gbest_position, w, c1, c2, num_plates):
        r1 = np.random.rand(num_plates)
        r2 = np.random.rand(num_plates)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, num_positions):
        self.position = np.clip(self.position + self.velocity, 0, num_positions - 1).astype(int)



class PSO_with_Batch:
    def __init__(self, num_particles, num_positions, num_plates, w, c1, c2, max_iter, lambda_1, lambda_2, lambda_3,
                 lambda_4,
                 dataset_name, objectives, use_adaptive):
        self.num_particles = num_particles
        self.num_positions = num_positions
        self.num_plates = num_plates  # 添加 num_plates 参数
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.use_adaptive = use_adaptive
        self.particles = [Particle(num_positions, num_plates) for _ in range(self.num_particles)]  # 传递 num_plates
        self.best_position = None
        self.best_score = np.inf
        self.worst_score = -np.inf
        self.best_improvement = 0
        self.total_improvement = 0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.convergence_data = []  # 收敛数据
        self.dataset_name = dataset_name
        self.stable_iterations = 0
        self.prev_best_score = np.inf
        self.start_time = None
        self.objectives = objectives
        self.adaptive_param_data = []

    def optimize(self):
        self.start_time = time.time()

        # 创建进度条
        progress_bar = st.progress(0)
        with st.spinner("Running PSO Optimization..."):
            for iteration in range(self.max_iter):
                improvement_flag = False
                for particle in self.particles:
                    # 计算当前粒子的得分
                    current_score = self.evaluate_particle(particle)

                    if current_score < particle.best_score:
                        particle.best_score = current_score
                        particle.best_position = particle.position.copy()

                    if current_score < self.best_score:
                        improvement_flag = True
                        self.best_improvement = max(self.best_improvement, self.best_score - current_score)
                        self.best_score = current_score
                        self.best_position = particle.position.copy()

                    if current_score > self.worst_score:
                        self.worst_score = current_score

                if improvement_flag:
                    self.total_improvement += self.prev_best_score - self.best_score
                    self.prev_best_score = self.best_score
                    self.stable_iterations = 0
                else:
                    self.stable_iterations += 1

                # 更新粒子的位置和速度
                for particle in self.particles:
                    particle.update_velocity(self.best_position, self.w, self.c1, self.c2, self.num_plates)
                    particle.update_position(self.num_positions)

                # 自适应调节
                if self.use_adaptive:
                    self.w, self.c1, self.c2 = apply_adaptive_pso(self.w, self.c1, self.c2,
                                                                  self.best_score - current_score,
                                                                  self.use_adaptive)
                    self.record_adaptive_params()

                # 更新收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])

                # 更新进度条
                progress_percentage = (iteration + 1) / self.max_iter
                progress_bar.progress(progress_percentage)

        # 优化结束后清除进度条
        progress_bar.empty()

        time_elapsed = time.time() - self.start_time
        self.save_metrics(time_elapsed)

        # 优化结束后，保存历史收敛数据
        history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "PSO")
        save_convergence_history(self.convergence_data, "PSO", self.dataset_name, history_data_dir)

    def evaluate_particle(self, particle):
        combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
            particle.position)
        energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(particle.position)
        balance_penalty = self.objectives.maximize_inventory_balance_v2(particle.position)
        space_utilization = self.objectives.maximize_space_utilization_v3(particle.position)

        score = (self.lambda_1 * combined_movement_turnover_penalty +
                 self.lambda_2 * energy_time_penalty +
                 self.lambda_3 * balance_penalty -
                 self.lambda_4 * space_utilization)
        return score

    def record_adaptive_params(self):
        self.adaptive_param_data.append({'w': self.w, 'c1': self.c1, 'c2': self.c2})

    def save_metrics(self, time_elapsed):
        iterations = len(self.convergence_data)
        save_performance_metrics(
            self.best_score, self.worst_score, self.best_improvement, self.total_improvement,
            iterations, time_elapsed, self.convergence_data, self.stable_iterations, self.dataset_name, "PSO"
        )