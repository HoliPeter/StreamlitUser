import numpy as np
import pandas as pd
import os
import streamlit as st
from datetime import datetime
from optimizers.pso_optimizer import PSO_with_Batch  # 引入 PSO_with_Batch
from optimizers.psosa_optimizer import PSO_SA_Optimizer  # PSO_SA 优化算法
from optimizers.eda_optimizer import EDA_with_Batch  # EDA 优化算法
from optimizers.ga_optimizer import GA_with_Batch  # 引入 GA 优化算法
from optimizers.co_ea_optimizer import CoEA_with_Batch  # 引入 CoEA 优化算法
from optimizers.sa_optimizer import SA_with_Batch  # 引入 SA_with_Batch
from utils import save_and_visualize_results, generate_stacking_distribution_statistics, generate_stacking_heatmaps, show_stacking_distribution_statistics, show_stacking_height_distribution_chart, add_download_button

class OptimizerRunner:
    def __init__(self, algorithms_params, df, area_positions, output_dir_base, flag=1):
        """
        初始化类，指定优化算法、数据集、库区分布和输出路径
        :param algorithms_params: 各种算法及其参数的字典
        :param df: 待优化的数据集
        :param area_positions: 库区的分布情况
        :param output_dir_base: 输出文件保存路径
        :param flag: 控制优化策略，1 表示选取最佳初始解后使用 SA 优化，2 表示分别运行各算法并选择最佳结果
        """
        self.algorithms_params = algorithms_params
        self.df = df
        self.area_positions = area_positions
        self.output_dir_base = output_dir_base
        self.results = []
        self.flag = flag

    def run_optimization(self):
        """
        运行所有指定的优化算法，并根据评分选择最佳结果
        """
        print("### 运行多种优化算法...")

        # 记录优化开始时间
        start_time = datetime.now()
        optimizer_runtime = {}  # 用于记录每个优化器的运行时间

        optimizer_classes = {
            "PSO_with_Batch": PSO_with_Batch,
            "PSO_SA_Optimizer": PSO_SA_Optimizer,  # 添加 PSO_SA_Optimizer
            "EDA_with_Batch": EDA_with_Batch,
            "GA_with_Batch": GA_with_Batch,
            "CoEA_with_Batch": CoEA_with_Batch
        }

        if self.flag == 0:
            # Flag=0: 只运行 PSO_SA_Optimizer 进行普通优化
            print("### 运行 PSO_SA_Optimizer 进行普通优化...")
            optimizer_name = "PSO_SA_Optimizer"
            optimizer_params = self.algorithms_params[optimizer_name]
            optimizer_class = optimizer_classes[optimizer_name]

            algo_start_time = datetime.now()

            optimizer = optimizer_class(**optimizer_params)
            optimizer.optimize()

            # 记录当前算法的运行时间
            algo_end_time = datetime.now()
            runtime = algo_end_time - algo_start_time
            optimizer_runtime[optimizer_name] = runtime
            print(f"### {optimizer_name} 运行时间为：{runtime}")
            print(f"### {optimizer_name} 最佳得分为：{optimizer.best_score:.2e}")

            # 保存和可视化结果
            output_file_plates_with_batch = save_and_visualize_results(
                optimizer, self.df, self.area_positions, self.output_dir_base, optimizer_name)

            # 存储优化结果
            self.results.append({
                "optimizer_name": optimizer_name,
                "best_score": optimizer.best_score,
                "output_file": output_file_plates_with_batch
            })



        elif self.flag == 1:
            # Flag = 1: 先使用多种算法选出最佳初始解，再用 SA 继续优化
            print("### 先使用多种算法运行并选取最佳初始解，然后使用 SA 进一步优化...")
            initial_results = []

            # 使用 PSO、GA、EDA、CoEA 选出初始解
            for optimizer_name in ["PSO_with_Batch", "EDA_with_Batch", "GA_with_Batch", "CoEA_with_Batch"]:
                optimizer_params = self.algorithms_params.get(optimizer_name)
                optimizer_class = optimizer_classes.get(optimizer_name)

                if not optimizer_class or not optimizer_params:
                    print(f"未找到 {optimizer_name} 的优化器类或参数，跳过此优化器。")
                    continue

                print(f"### 正在运行 {optimizer_name} ...")
                algo_start_time = datetime.now()

                optimizer = optimizer_class(**optimizer_params)
                optimizer.optimize()

                # 记录当前算法的运行时间
                algo_end_time = datetime.now()
                runtime = algo_end_time - algo_start_time
                optimizer_runtime[optimizer_name] = runtime
                print(f"### {optimizer_name} 运行时间为：{runtime}")
                print(f"### {optimizer_name} 最佳得分为：{optimizer.best_score:.2e}")  # 输出评分

                # 保存结果用于比较
                initial_results.append({
                    "optimizer_name": optimizer_name,
                    "best_score": optimizer.best_score,
                    "best_position": optimizer.best_position  # 统一使用best_position
                })

            # 选择初始解中评分最低的结果
            best_initial_result = min(initial_results, key=lambda x: x["best_score"])
            print(
                f"### 最佳初始解为：{best_initial_result['optimizer_name']}，得分为：{best_initial_result['best_score']:.2e}")

            # 使用 SA 进行进一步优化
            print("### 使用 SA 进行进一步优化...")
            sa_params = self.algorithms_params["SA_with_Batch"]
            sa_params["initial_solution"] = best_initial_result["best_position"]

            optimizer = SA_with_Batch(**sa_params)
            optimizer.optimize()

            # 记录 SA 运行时间
            sa_end_time = datetime.now()
            sa_runtime = sa_end_time - algo_end_time
            optimizer_runtime["SA_with_Batch"] = sa_runtime
            print(f"### SA 运行时间为：{sa_runtime}")
            print(f"### SA 最佳得分为：{optimizer.best_score:.2e}")  # 输出SA评分

            # 保存最终结果
            self.results.append({
                "optimizer_name": "SA_with_Batch",
                "best_score": optimizer.best_score,
                "output_file": save_and_visualize_results(optimizer, self.df, self.area_positions, self.output_dir_base,
                                                          "SA_with_Batch")
            })

        elif self.flag == 2:
            # Flag = 2: 按原来的方式运行多种算法并选择最佳结果
            print("### 分别运行各算法并选择最佳结果...")
            for optimizer_name, optimizer_params in self.algorithms_params.items():
                optimizer_class = optimizer_classes.get(optimizer_name)
                if not optimizer_class:
                    print(f"未找到 {optimizer_name} 的优化器类，跳过此优化器。")
                    continue

                print(f"### 正在运行 {optimizer_name} ...")
                algo_start_time = datetime.now()

                optimizer = optimizer_class(**optimizer_params)
                optimizer.optimize()

                # 记录当前算法的运行时间
                algo_end_time = datetime.now()
                runtime = algo_end_time - algo_start_time
                optimizer_runtime[optimizer_name] = runtime
                print(f"### {optimizer_name} 运行时间为：{runtime}")
                print(f"### {optimizer_name} 最佳得分为：{optimizer.best_score:.2e}")  # 输出评分

                # 保存和可视化结果
                output_file_plates_with_batch = save_and_visualize_results(optimizer, self.df, self.area_positions, self.output_dir_base, optimizer_name)

                # 存储每个优化器的最佳分数及其结果
                self.results.append({
                    "optimizer_name": optimizer_name,
                    "best_score": optimizer.best_score,
                    "output_file": output_file_plates_with_batch
                })

        # 选择最佳的优化结果
        self.select_best_result()

        # 记录优化结束时间并计算总运行时间
        end_time = datetime.now()
        total_runtime = end_time - start_time
        print("\n### 各优化算法运行时间：")
        for optimizer_name, runtime in optimizer_runtime.items():
            print(f"{optimizer_name}：{runtime}")
        print(f"\n### 总共运行时间为：{total_runtime}")

    def select_best_result(self):
        """
        比较所有优化器的结果，选择最佳结果进行后续处理
        """
        if not self.results:
            print("未找到任何可用的优化结果。")
            return

        # 找到最佳结果（评分最低的）
        best_result = min(self.results, key=lambda x: x["best_score"])
        print(f"### 最佳优化器为：{best_result['optimizer_name']}，最佳得分为：{best_result['best_score']:.2e}")

        # 只展示最佳优化算法的垛位分布统计表和堆垛高度分布
        result_df, all_positions, all_heights = generate_stacking_distribution_statistics(
            self.df, self.area_positions, self.output_dir_base, best_result['optimizer_name'])

        # 先展示垛位分布统计表
        show_stacking_distribution_statistics(result_df, best_result['optimizer_name'], self.output_dir_base)

        # 再展示堆垛高度分布
        show_stacking_height_distribution_chart(all_positions, all_heights, best_result['optimizer_name'])

        # 生成库区堆垛俯视图
        print("### 库区堆垛俯视图 - 最佳结果")
        generate_stacking_heatmaps(self.df, self.area_positions)

        # 提供下载按钮
        add_download_button(best_result['output_file'], best_result['optimizer_name'])

