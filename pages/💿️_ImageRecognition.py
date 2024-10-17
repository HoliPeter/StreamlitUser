import os
import easyocr
import pandas as pd
import streamlit as st
from backend.Recognition_processing import ImgRec
from backend import constants
from backend import utils as ut
import plotly.graph_objects as go
import time


# 加载 OCR 模型
@st.cache_resource
def load_ocr_model():
    # 使用 st.spinner 来显示加载提示
    with st.spinner("模型加载中......"):
        reader = easyocr.Reader(['en'], model_storage_directory="data/easyorc_models")
        CodeRecognizer = ImgRec()
        CodeRecognizer.set_reader(reader)
    return CodeRecognizer


code_recognizer = load_ocr_model()

# st.title("💿️ 钢 板 编 码 识 别")

# 使用 HTML 和 CSS 设置标题居中和字体
st.markdown(
    """
    <h1 style='text-align: center; font-family: SimSun;'>
        💿️ 钢板编码识别
    </h1>
    """,
    unsafe_allow_html=True
)

col01,col02=st.columns([0.2,0.8])
with col01:
    # 选择识别模式
    mode = st.selectbox('请选择识别模式（📷/🎥）', ['图像识别 📷', '视频识别 🎥'])

if mode == '图像识别 📷':
    col03, col04,col05 = st.columns([0.2, 0.2, 0.6])
    with col03:
        option = st.selectbox('选择图片来源', ['测试数据集', '上传图片'])
    if option == '测试数据集':
        base_image_folder = constants.BASE_IMAGE_DIR
        subfolders = [f for f in os.listdir(base_image_folder) if
                      os.path.isdir(os.path.join(base_image_folder, f)) and f.startswith('Image_src')]
        if subfolders:
            # col05, col06 = st.columns([0.2, 0.8])
            with col04:
                selected_subfolder = ''
                selected_subfolder = st.selectbox(' 请选择一个图像文件夹', subfolders, key="key_for_ImgRec_folder")
            target_folder_path = os.path.join(base_image_folder, selected_subfolder)

            if st.button('🚀 开始识别'):
                placeholder = st.empty()
                placeholder.info('正在识别图像中的钢板编号...')
                with st.spinner('加载中，请稍候...'):
                    progress_placeholder = st.empty()  # 识别进度条
                    # 进行识别并返回结果进行显示
                    total_images = code_recognizer.process_images_from_folder(target_folder_path, progress_placeholder)
                    progress_placeholder.empty()
                    if total_images == 0:
                        placeholder.warning(f'文件夹 {selected_subfolder} 中未找到任何图像！')
                    else:
                        placeholder.success(
                            f' 识别完成！结果已保存到 recognized_results.csv （数据集：{selected_subfolder}）')


    elif option == '上传图片':
        uploaded_files = st.file_uploader('上传图像文件', type=['jpg', 'png', 'bmp'], accept_multiple_files=True)
        if uploaded_files:
            if st.button('🚀 Start Recognition'):
                placeholder = st.empty()
                placeholder.info('正在识别图像中的钢板编号...')
                with st.spinner('加载中，请稍候...'):
                    progress_placeholder = st.empty()  # 识别进度条
                    # 进行识别并返回结果进行显示
                    code_recognizer.process_uploaded_images(uploaded_files, progress_placeholder)
                    progress_placeholder.empty()
                    placeholder.success(f'识别完成！结果已保存到 recognized_results.csv ')

elif mode == '视频识别 🎥':
    videos = [f for f in os.listdir(constants.BASE_VIDEO_DIR) if f.endswith(('.mp4', '.avi'))]
    selected_video = st.selectbox('🎬 请选择视频文件', videos, key="key_for_VidRec_file")
    frame_interval = st.number_input("选择帧间隔", min_value=1, value=20)
    if st.button('开始识别'):
        placeholder = st.empty()
        placeholder.info('正在识别视频中的钢板编号...')
        with st.spinner('加载中，请稍候...'):
            progress_placeholder = st.empty()  # 识别进度条
            video_path = os.path.join(constants.BASE_VIDEO_DIR, selected_video)
            code_recognizer.process_video_from_folder(frame_interval, video_path, progress_placeholder)
            progress_placeholder.empty()
            placeholder.success(f' 识别完成！结果已保存到 recognized_results.csv ')





# 最新识别结果显示
st.markdown("<h5 style='text-align: left; color: black;'> 最新识别结果：</h5>", unsafe_allow_html=True)

if code_recognizer.current_df is not None:
    df = code_recognizer.current_df
    total_rows = df.shape[0]
    page_size = 10  # 每页显示10行
    total_pages = (total_rows + page_size - 1) // page_size  # 计算总页数

    # 分页显示数据集
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
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

    # 图表显示
    st.markdown("<h5 style='text-align: left; color: black;'> 数据图表：</h5>", unsafe_allow_html=True)

    # 生成简单的柱状图，显示钢板的数量分布（可根据实际数据列调整）
    if 'Material_Code' in df.columns:
        material_counts = df['Material_Code'].value_counts()
        st.bar_chart(material_counts)

    # 生成其他图表（如堆垛高度分布）
    if 'Height' in df.columns:
        st.line_chart(df['Height'])

else:
    st.write('暂无识别结果')


# 在侧边栏添加一个复选框
toggle_state = st.checkbox("显示详细数据")  # 复选框，类似开关
# 根据复选框状态显示不同的内容
if toggle_state:
    # 显示csv，并提供下载
    csv_file_path = constants.CSV_FILE_PATH
    csv_other_path = constants.CSV_OTHER_PATH
    result_file_path = 'result/ImageRecognition_CSV/Output_steel_data.csv'

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'>📄 结果 CSV 文件内容：</h5>", unsafe_allow_html=True)

    # 自定义 CSS，设置表格缩放
    st.markdown("""
                        <style>
                        .scaled-table {
                            transform: scale(0.8); /* 按比例缩放表格 */
                            transform-origin: top left; /* 缩放原点 */
                        }
                        </style>
                        """, unsafe_allow_html=True)

    # 创建两个列
    col_download, col_clear = st.columns([0.5, 0.5])

    # 处理清除 CSV 内容的逻辑
    with col_clear:
        # 显示识别结果（CSV 表格）
        if os.path.exists(csv_file_path):
            if ut.is_csv_empty(csv_file_path):  # 检查 CSV 是否为空
                st.warning('⚠️ 没有可用的识别数据')
            else:
                # 清除识别结果（CSV 表格）
                if st.button('🗑 清除 CSV 文件内容'):
                    with st.spinner('正在清除 CSV 文件内容...'):
                        try:
                            ut.clear_csv(csv_file_path)  # 调用自定义的清除 CSV 文件内容的函数
                            ut.clear_csv(csv_other_path)
                            st.success(' CSV 文件内容已清除')
                        except Exception as e:
                            st.error(f" 清除 CSV 文件时出错: {e}")
                if not ut.is_csv_empty(csv_file_path):
                    df = pd.read_csv(csv_file_path)
                    # 使用缩小比例显示DataFrame
                    st.markdown('<div class="scaled-table">', unsafe_allow_html=True)
                    st.dataframe(df)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning('⚠️ 没有可用的识别数据')
        else:
            st.warning('⚠️ CSV 文件不存在。')
    # 处理下载 CSV 的逻辑
    with col_download:
        if os.path.exists(csv_file_path):  # 检查CSV文件路径是否存在
            if ut.is_csv_empty(csv_file_path):  # 检查CSV文件是否为空
                st.warning('⚠️ 没有可用的识别数据')  # 如果为空，提示警告信息
            else:
                # 读取两个CSV文件
                df1 = pd.read_csv(csv_file_path)
                df2 = pd.read_csv(csv_other_path)

                # 检查必要的列是否存在
                if "Recognized Text" in df1.columns and "Filename" in df1.columns and \
                        "Filename" in df2.columns and "Entry Time" in df2.columns and "Delivery Time" in df2.columns and "Batch" in df2.columns:

                    # 调用自定义函数生成新的CSV文件
                    result_df = ut.generate_csv_from_column(df1, df2, "Recognized Text")

                    # 保存结果到指定文件
                    result_df.to_csv(result_file_path, index=False)

                    # 转换DataFrame为CSV格式并编码为UTF-8
                    csv = result_df.to_csv(index=False).encode('utf-8')

                    # 确保生成的CSV内容存在后再显示下载按钮
                    if csv:
                        st.download_button(
                            label="下载处理后的CSV文件",  # 按钮的标签
                            data=csv,  # 下载的数据
                            file_name='Output_steel_data.csv',  # 下载文件的名称
                            mime='text/csv',  # 文件的MIME类型
                        )

                    # 显示处理后的DataFrame
                    st.dataframe(result_df)
                else:
                    # 如果必要的列不存在，显示错误提示信息
                    if "Recognized Text" not in df1.columns:
                        st.error("CSV文件中没有找到 'Recognized Text' 列")
                    if "Filename" not in df1.columns:
                        st.error("CSV文件中没有找到 'Filename' 列")
        else:
            st.warning("⚠️ CSV 文件不存在。")  # 如果文件不存在，提示警告信息

    # 显示柱状图
    csv_file_path = constants.CSV_FILE_PATH

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'> 历史汇总： </h5>", unsafe_allow_html=True)

    # 检查 CSV 文件是否存在
    if not os.path.exists(csv_file_path):
        st.error(" 错误：CSV 文件不存在。")
        st.stop()

    if ut.is_csv_empty(csv_file_path):
        st.warning("⚠️ CSV 文件为空。")
        st.stop()

    # 读取 CSV 文件
    try:
        data = pd.read_csv(csv_file_path)

    except Exception as e:
        st.error(f"❌ 读取 CSV 文件时出错: {e}")
        st.stop()

    # 检查是否有所需的列
    required_columns = ['Filename', 'Average Confidence', 'Timestamp', 'Accuracy']
    if not all(column in data.columns for column in required_columns):
        st.error("❌ 错误：CSV 文件缺少 'Filename', 'Average Confidence' 或 'Timestamp' 列。")
        st.stop()

    # 合并 Filename 和 Timestamp 作为 x 轴标签
    x_labels = data['Filename'] + ' ' + data['Timestamp']
    average_confidences = data['Average Confidence'].tolist()
    # 转换 'Accuracy' 列为小数
    accuracy = data['Accuracy'].str.rstrip('%').astype('float') / 100
    # 设置保存图表的目录
    save_dir = "result/Historical_barChart"
    ut.ensure_directory_exists(save_dir)

    # 设置固定颜色为浅蓝色
    bar_color = '#ADD8E6'  # 浅蓝色

    # 创建水平柱状图
    fig1 = go.Figure(data=[
        go.Bar(y=x_labels, x=average_confidences, marker_color=bar_color, orientation='h')
    ])

    fig1.update_layout(
        title="每个文件及时间戳的平均置信度",
        yaxis_title="文件名 + 时间戳",
        xaxis_title="平均置信度",
        hoverlabel=dict(
            bgcolor="white",
            font_color="black"
        )
    )


    # 获取柱子的宽度，根据元素数量动态调整
    def get_bar_width(num_positions):
        if num_positions <= 3:
            return 0.3
        elif num_positions <= 6:
            return 0.2
        else:
            return 0.1


    bar_width = get_bar_width(len(x_labels))

    # 创建组合图
    fig2 = go.Figure()

    # 添加柱状图
    fig2.add_trace(go.Bar(
        x=x_labels,
        y=accuracy,
        name='柱状图',
        width=[bar_width] * len(x_labels),
        marker_color='lightblue',  # 使用与之前相同的颜色
        hovertemplate='%{x}, %{y:.2%}',
    ))

    # 添加折线图
    fig2.add_trace(go.Scatter(
        x=x_labels,
        y=accuracy,
        mode='lines+markers',
        name='折线图'
    ))

    # 设置图表布局
    fig2.update_layout(
        title=" 每个文件及时间戳的准确度率 - 组合图",
        xaxis_title="文件名 + 时间戳",
        yaxis_title="准确率",
        hoverlabel=dict(
            bgcolor="white",
            font_color="black"
        ),
        xaxis_tickangle=-45,  # 将 x 轴标签旋转以防止重叠
        yaxis=dict(tickformat=".2%")  # 将 y 轴刻度格式化为百分比
    )

    # 显示图表
    st.plotly_chart(fig2, use_container_width=True)
    # 显示图表
    st.plotly_chart(fig1)


else:
    st.write('详细数据已隐藏')
