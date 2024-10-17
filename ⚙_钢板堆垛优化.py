import streamlit as st
import pandas as pd
import numpy as np
import os
from optimization_objectives import SteelPlateStackingObjectives as OptimizationObjectives
from optimizers.psosa_optimizer import PSO_SA_Optimizer
from utils import save_convergence_history, add_download_button, run_optimization, display_icon_with_header,display_icon_with_selectbox
from optimizer_runner import OptimizerRunner  # 导入优化算法管理器
import time

# 设定默认的图片文件夹路径
image_folder_path = "data/introduction_src/images01"
icon_path = "data/icon/icon02.jpg"

# 从 constants 文件中引入常量
from constants import (
    OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR, TEST_DATA_PATH,
    AREA_POSITIONS_DIMENSIONS,
    DEFAULT_AREA_POSITIONS, DEFAULT_STACK_DIMENSIONS,
    HORIZONTAL_SPEED, VERTICAL_SPEED, STACK_FLIP_TIME_PER_PLATE,
    INBOUND_POINT, OUTBOUND_POINT, Dki
)

# Streamlit 页面配置
st.set_page_config(page_title="智能钢板堆垛系统", page_icon="⚙", layout="wide")


# 创建用于保存图像的目录
output_dir_base = "result/"
os.makedirs(output_dir_base, exist_ok=True)

# st.title("⚙ 智能钢板堆垛系统")
# 使用 HTML 和 CSS 设置标题居中和字体
st.markdown(
    """
    <h1 style='text-align: center; font-family: SimSun;'>
        ⚙ 智能钢板堆垛系统
    </h1>
    """,
    unsafe_allow_html=True
)

col021,col22,col23= st.columns([0.2,0.2,0.6])
with col021:

    use_default_config = st.checkbox("使用默认库区配置", value=True)

# 检查是否使用默认配置
if not use_default_config:
    # 使用下拉框选择当前配置的库区
    area_positions = {}
    stack_dimensions = {}

    # 如果用户选择自定义配置，显示相关输入框
    num_areas = st.number_input("请输入库区数量", 1, 10, 6)

    # 初始化各个库区的配置
    for area in range(num_areas):
        area_positions[area] = []
        stack_dimensions[area] = []

    selected_area = st.selectbox(
        "选择要配置的库区",
        [f"库区 {i + 1}" for i in range(num_areas)],
        key="selected_area"
    )
    area_index = int(selected_area.split(" ")[1]) - 1

    # 输入当前库区的垛位数量
    num_stacks = st.number_input(f"请输入 {selected_area} 的垛位数量", 1, 10, 4, key=f'num_stacks_area_{area_index}')

    # 垛位配置
    area_stack_positions = []
    area_stack_dimensions = []

    for stack in range(num_stacks):
        # 三列布局输入垛位的坐标和尺寸
        cols = st.columns([0.3, 0.3, 0.4])
        col13, col14, col15 = cols  # 解包列对象
        with col13:
            x = st.number_input(f"垛位 {stack + 1} 的 X 坐标", key=f'stack_x_area_{area_index}_{stack}')
        with col14:
            y = st.number_input(f"垛位 {stack + 1} 的 Y 坐标", key=f'stack_y_area_{area_index}_{stack}')
        with col13:
            width = st.number_input(f"垛位 {stack + 1} 的宽度（毫米）", 1000, 20000, 6000,
                                    key=f'stack_width_area_{area_index}_{stack}')
        with col14:
            length = st.number_input(f"垛位 {stack + 1} 的长度（毫米）", 1000, 20000, 3000,
                                     key=f'stack_length_area_{area_index}_{stack}')

        # 保存当前垛位的配置
        area_stack_positions.append((x, y))
        area_stack_dimensions.append((length, width))

    # 更新当前库区的配置
    area_positions[area_index] = area_stack_positions
    stack_dimensions[area_index] = area_stack_dimensions

else:
    # 使用默认配置
    area_positions = DEFAULT_AREA_POSITIONS
    stack_dimensions = DEFAULT_STACK_DIMENSIONS


with col22:
    if st.button("🗄️查看配置"):
        st.session_state["show_stack_config"] = not st.session_state["show_stack_config"]

# 查看当前配置
if "show_stack_config" not in st.session_state:
    st.session_state["show_stack_config"] = False


# 使用 st.empty() 占位符来显示提示信息
info_placeholder = st.empty()

if st.session_state["show_stack_config"]:
    # 显示提示信息
    info_placeholder.info("提示：再次点击按钮可隐藏当前配置...")
    # 三秒后自动清除提示信息
    time.sleep(1)
    info_placeholder.empty()

if st.session_state["show_stack_config"]:
    st.write("### 当前库区配置")

    # 将区域位置和堆垛尺寸转换为 DataFrame 格式
    positions_data = []
    dimensions_data = []

    for area, positions in area_positions.items():
        for i, (x, y) in enumerate(positions):
            positions_data.append([f"区域 {area + 1}", f"垛位 {i + 1}", x, y])

    for area, dimensions in stack_dimensions.items():
        for i, (length, width) in enumerate(dimensions):
            dimensions_data.append([f"区域 {area + 1}", f"垛位 {i + 1}", length, width])

    # 创建 DataFrame
    positions_df = pd.DataFrame(positions_data, columns=["区域", "垛位", "X 坐标 (毫米)", "Y 坐标 (毫米)"])
    dimensions_df = pd.DataFrame(dimensions_data, columns=["区域", "垛位", "长度 (毫米)", "宽度 (毫米)"])

    # 显示表格
    cols = st.columns([0.4, 0.4, 0.2])
    col011, col012, col013 = cols

    with col011:
        st.write("#### 区域位置")
        st.dataframe(positions_df)

    with col012:
        st.write("#### 堆垛尺寸")
        st.dataframe(dimensions_df)

    # 显示表格
    col014, col015, col016 = st.columns([0.3, 0.3, 0.4])
    with col014:
        st.write("#### 库区图片")
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        selected_image = st.selectbox("选择库区图片", ["请选择"] + image_files)

        # 设置图片的显示大小
        max_image_width = 600  # 设置图片的最大宽度，单位为像素

        # 显示选中的图片
        if selected_image != "请选择":
            image_path = os.path.join(image_folder_path, selected_image)
            st.image(image_path, caption=f"库区图片 - {selected_image}", width=max_image_width)



# 使用 display_icon_with_header 函数替换现有的图标和标题显示逻辑
display_icon_with_header("data/icon/icon01.jpg", "数据导入", font_size="24px", icon_size="20px")

# 使用 display_icon_with_header 函数替换部分的展示
col3, col4 = st.columns([0.3, 0.7])
with col3:
    options = ["使用系统数据集", "上传自定义数据集"]
    data_choice = display_icon_with_selectbox(icon_path, "选择数据集", options)



df = None
dataset_name = None
system_data_dir = "data/Steel_Data"


# 检查是否选择了数据集
data_selected = False

# 导入数据集的逻辑
if data_choice == "上传自定义数据集":
    uploaded_file = st.file_uploader("上传钢板数据集 (CSV)", type=["csv"])
    if uploaded_file:
        dataset_name = uploaded_file.name.split('.')[0]
        df = pd.read_csv(uploaded_file)
        st.write("已上传的数据集：")
        st.write(df.head())
        data_selected = True  # 表示数据已选择
    else:
        st.warning("请上传数据集以继续。")
elif data_choice == "使用系统数据集":
    # 使用自定义函数创建选择数据集和优化模式的布局
    col5, col6, col7 = st.columns([0.3, 0.3, 0.4])

    with col5:
        # 左侧：选择数据集
        available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]
        selected_dataset = display_icon_with_selectbox(
            "data/icon/icon02.jpg",
            "选择系统数据集",
            [""] + available_datasets,
            key="dataset_selectbox"
        )
        if selected_dataset:
            dataset_name = selected_dataset
            system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")
            df = pd.read_csv(system_dataset_path)
            data_selected = True  # 表示数据已选择

    with col6:
        # 右侧：选择优化模式
        optimization_mode = display_icon_with_selectbox(
            "data/icon/icon02.jpg",
            "选择优化模式",
            ["普通优化", "深度优化"],
            key="optimization_mode_selectbox"
        )


import time

# 初始化显示/隐藏数据集的状态
if "show_dataset" not in st.session_state:
    st.session_state["show_dataset"] = False

# 创建按钮布局
col1, col2 = st.columns([0.3, 0.7])

with col1:
    # 左侧：查看训练数据集按钮
    view_dataset = st.button("🗳️ 查看数据集")

    # 切换显示/隐藏状态
    if view_dataset:
        st.session_state["show_dataset"] = not st.session_state["show_dataset"]
        # 显示提示信息，并在2秒钟后自动消失
        placeholder = st.empty()
        if st.session_state["show_dataset"]:
            placeholder.info("再次点击按钮可隐藏训练数据集")

        time.sleep(1)  # 等待2秒
        placeholder.empty()  # 清空提示信息

with col2:
    # 右侧：开始优化按钮
    start_work = st.button("🔧 开始优化")

# 如果需要显示数据集
if st.session_state["show_dataset"]:
    if 'df' in locals() and data_selected:
        st.write("#### 训练数据集预览")

        # 分页显示数据集
        page_size = 10  # 每页显示10行
        total_rows = df.shape[0]
        total_pages = (total_rows + page_size - 1) // page_size  # 计算总页数

        col25,col26=st.columns([0.5,0.5])

        with col25:
            # 选择页码
            page_num = st.number_input(
                f"选择页码 (共 {total_pages} 页)",
                min_value=1,
                max_value=total_pages,
                step=1,
                value=1,
                format="%d",
            )

        # 计算当前页数据的起始和结束索引
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        # 显示当前页的数据
        st.dataframe(df.iloc[start_idx:end_idx])

        # 显示当前页码和总页数
        st.write(f"当前显示第 {page_num} 页，共 {total_pages} 页")
    else:
        st.warning("尚未选择数据集，请先选择数据集。")

# 在没有选择数据集的情况下，点击“开始优化”按钮时的提示
if start_work and not ('df' in locals() and data_selected):
    st.warning("请先选择训练数据集后再开始优化。")




if start_work:
    if not data_selected:
        # 没有选择数据集时显示提示信息，并在三秒后自动消失
        warning_placeholder = st.empty()
        warning_placeholder.warning("请先选择数据集后再继续优化...")
        time.sleep(3)
        warning_placeholder.empty()


# 优化参数配置
initial_temperature = 1000.0
cooling_rate = 0.9
min_temperature = 0.1
max_iterations_sa = 5
num_particles = 30  # 粒子群大小
max_iter_pso = 1  # PSO最大迭代次数
w, c1, c2 = 0.5, 1.5, 1.5  # PSO 参数
lambda_1, lambda_2, lambda_3, lambda_4 = 1.0, 1.0, 1.0, 1.0
use_adaptive = True

# EDA 优化参数配置
pop_size = 50  # EDA 种群大小
max_iter_eda = 1  # EDA最大迭代次数
mutation_rate = 0.1  # EDA变异率
crossover_rate = 0.7  # EDA交叉率

# GA 优化参数配置
ga_population_size = 50
ga_generations = 1  # GA最大迭代次数
ga_mutation_rate = 0.1
ga_crossover_rate = 0.8

# CoEA 优化参数配置
coea_population_size = 50  # CoEA 种群大小
coea_generations = 1  # CoEA 最大迭代次数
coea_mutation_rate = 0.1  # CoEA 变异率
coea_crossover_rate = 0.8  # CoEA 交叉率

# 优化分析
if df is not None:
    output_dir = os.path.join(output_dir_base, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # 数据准备
    plates = df[['Length', 'Width', 'Thickness', 'Material_Code', 'Batch', 'Entry Time', 'Delivery Time']].values
    num_positions = len(Dki)
    num_plates = len(plates)
    heights = np.zeros(num_positions)
    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values
    batches = df['Batch'].values

    objectives = OptimizationObjectives(
        plates=plates,
        heights=heights,
        delivery_times=delivery_times,
        batches=batches,
        Dki=Dki,
        area_positions=DEFAULT_AREA_POSITIONS,
        inbound_point=INBOUND_POINT,
        outbound_point=OUTBOUND_POINT,
        horizontal_speed=HORIZONTAL_SPEED,
        vertical_speed=VERTICAL_SPEED
    )

    # 多种优化算法的参数配置
    algorithms_params = {
        "PSO_with_Batch": {
            'num_particles': num_particles,
            'num_positions': num_positions,
            'num_plates': num_plates,  # 添加 num_plates 参数
            'w': w,
            'c1': c1,
            'c2': c2,
            'max_iter': max_iter_pso,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        },
        "SA_with_Batch": {
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations': max_iterations_sa,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': num_positions,
            'num_plates': num_plates,
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        },
        "PSO_SA_Optimizer": {
            'num_particles': num_particles,
            'num_positions': num_positions,
            'w': w,
            'c1': c1,
            'c2': c2,
            'max_iter_pso': max_iter_pso,
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations_sa': max_iterations_sa,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_plates': num_plates,  # 钢板数量
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        },
        "EDA_with_Batch": {
            'pop_size': pop_size,
            'num_positions': num_positions,
            'num_plates': num_plates,  # 添加 num_plates 参数
            'max_iter': max_iter_eda,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        },
        "GA_with_Batch": {
            'population_size': ga_population_size,
            'mutation_rate': ga_mutation_rate,
            'crossover_rate': ga_crossover_rate,
            'generations': ga_generations,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': num_positions,
            'dataset_name': dataset_name,
            'objectives': objectives,
            'plates': plates,
            'delivery_times': delivery_times,
            'batches': batches,
            'use_adaptive': use_adaptive
        },
        "CoEA_with_Batch": {
            'population_size': coea_population_size,
            'mutation_rate': coea_mutation_rate,
            'crossover_rate': coea_crossover_rate,
            'generations': coea_generations,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': num_positions,
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        }
    }

    if start_work:
        # 使用 OptimizerRunner 进行优化
        if optimization_mode == "深度优化":  # 判断用户是否选择深度优化模式
            # 启用深度优化，运行多个优化算法并选择最佳方案，Flag=2 表示多种算法分别运行
            optimizer_runner = OptimizerRunner(algorithms_params, df, DEFAULT_AREA_POSITIONS, output_dir_base, flag=1)
            optimizer_runner.run_optimization()
        else:
            # 普通优化，Flag=0 表示只使用单一算法 PSO_SA_Optimizer
            optimizer_runner = OptimizerRunner(algorithms_params, df, DEFAULT_AREA_POSITIONS, output_dir_base, flag=0)
            optimizer_runner.run_optimization()

