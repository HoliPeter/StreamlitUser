import os
import re
import cv2
import shutil
import imagehash
import pandas as pd
from PIL import Image
from backend import constants


#csv是否为空
def is_csv_empty(file_path):
    """检查CSV文件是否为空"""
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0  # 文件存在且大小为0时返回True
# 创建文件夹（如果不存在）
def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
#清空图片文件夹
def clear_folder(folder_path):
    """清空图片文件夹"""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)  # 删除文件夹中的所有图片
#清空CSV文件内容
def clear_csv(file_path):
    """清空CSV文件内容"""
    if os.path.exists(file_path):
        open(file_path, 'w').close()  # 打开并清空文件内容
#保存数据到csv
def append_to_csv(data, file_path):
    """将识别结果追加到CSV文件中"""
    df = pd.DataFrame(data)  # 将数据转换为DataFrame格式
    # 如果文件存在，则追加数据；否则创建新文件并写入表头
    if os.path.exists(file_path):
        if is_csv_empty(file_path):
            df.to_csv(file_path, mode='a', header=True, index=False)  # 追加模式，不写入表头
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)  # 追加模式，不写入表头
    else:
        df.to_csv(file_path, index=False)  # 如果文件不存在，创建文件并写入表头
        df.to_csv(file_path, mode='a', header=False, index=False)  # 再次追加数据以防止错误
#计算字符串准确率
def calculate_accuracy(recognized_text, correct_text):
    # 去掉识别文本和正确文本中的空格用于匹配
    recognized_text = recognized_text.replace(" ", "")
    correct_text = correct_text.replace(" ", "").replace("\n", "")

    # 获取最小的长度，防止索引越界
    min_len = min(len(recognized_text), len(correct_text))

    # 统计匹配的字符个数
    match_count = sum(1 for i in range(min_len) if recognized_text[i] == correct_text[i])

    # 准确率 = 匹配字符数 / 正确编码的总长度
    accuracy = match_count / len(correct_text) if len(correct_text) > 0 else 0
    return accuracy
#字符串校准
def process_steel_code(input_str):
    # 去除字符串末尾的空格
    input_str = input_str.rstrip()

    # 找到最后一个空格
    last_space_idx = input_str.rfind(' ')

    # 分离字符串为两部分：空格前和空格后的内容
    if last_space_idx != -1:
        # 空格前的部分保持不变
        before_space = input_str[:last_space_idx + 1]
        # 空格后的部分进行替换操作
        after_space = input_str[last_space_idx + 1:]
        #print(f"as1:{(after_space)}")

        # 替换大小写的'O'为'0'，大小写的'X'为'*'
        before_space = before_space.replace('GBIT', 'GB/T')
        after_space = after_space.replace('O', '0').replace('o', '0')
        after_space = after_space.replace('X', '*').replace('x', '*')
        after_space = after_space.replace('I', '1').replace('|', '1')
        after_space = after_space.replace('Z', '2').replace('z', '2')
        #print(f"as2:{after_space}")
        # 返回替换后的完整字符串
        return before_space + after_space
    else:
        return input_str
#从视频中截取帧，保存并去重
def extract_unique_frames_from_video(frame_interval, video_path):
    frames_cache_folder = constants.frames_cache_folder
    frames_final_folder = constants.frames_final_folder
    # 创建文件夹（如果不存在）
    ensure_directory_exists(frames_cache_folder)
    ensure_directory_exists(frames_final_folder)
    #清除文件夹内容
    clear_folder(frames_cache_folder)
    clear_folder(frames_final_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0

    # 读取视频帧并保存图像
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(frames_cache_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"已保存{saved_frame_count}张图像到 {frames_cache_folder} 文件夹中。")

    # 设置哈希容差，容差越小，相似度要求越高
    hash_tolerance = 5
    hashes = []

    # 遍历源文件夹中的所有图像
    for filename in sorted(os.listdir(frames_cache_folder)):
        file_path = os.path.join(frames_cache_folder, filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = Image.open(file_path)

            # 计算图像的感知哈希值
            img_hash = imagehash.phash(image)

            # 检查哈希列表中是否存在相似的图像
            if all(abs(img_hash - existing_hash) > hash_tolerance for existing_hash in hashes):
                # 如果没有相似图像，将哈希值加入列表
                hashes.append(img_hash)
                # 保存该图像到目标文件夹
                shutil.copy(file_path, os.path.join(frames_final_folder, filename))
                print(f"保留图像: {filename}")

    clear_folder(frames_cache_folder)   #清除缓存图片
    print(f"去重完成，共保留 {len(hashes)} 张图像。")
#
def generate_csv_from_column(df1, df2, column_name):
    # 去除首尾空格，并将连续空格替换为一个空格
    def clean_string(input_string):
        return re.sub(r'\s+', ' ', input_string.strip())

    # 提取"Material"列
    def extract_material(input_string):
        match_material = re.search(r'Q\S*', input_string)
        return match_material.group(0) if match_material else None

    # 提取"Furnace"列
    def extract_furnace(input_string, material):
        furnace_split = input_string.split(' ')
        if material in furnace_split:
            index_after_material = furnace_split.index(material) + 2
            if index_after_material < len(furnace_split):
                return furnace_split[index_after_material]
        return None

    # 提取"Standard"列
    def extract_standard(input_string):
        # 正则表达式匹配 "GB/T" 开头，直到第二个空格之前
        match_standard = re.search(r'(GB/T[^\s]*\s[^\s]*)', input_string)
        if match_standard:
            return match_standard.group(0)
        return None

    # 提取"Thickness", "Length", "Width"列
    def extract_dimensions(input_string):
        # 使用正则表达式匹配维度：三个数字由 x, X 或 * 分隔
        match_dimensions = re.search(r'(\d+)[xX*](\d+)[xX*](\d+)', input_string)
        if match_dimensions:
            values = list(map(int, match_dimensions.groups()))
            # 对三个值进行排序，确保长 > 宽 > 厚
            values.sort(reverse=True)
            length, width, thickness = values
            return thickness, length, width

        return None, None, None


    # 假设 df1 和 df2 有相同的 Filename 列，通过 Filename 列合并两个 DataFrame
    merged_df = pd.merge(df1, df2[['Filename', 'Timestamp', 'Entry Time', 'Delivery Time', 'Batch']],
                         on=['Filename', 'Timestamp'], how='left')

    result_data = []  # 存储结果的列表

    # 对每一行进行分割
    num = 0
    for index, row in merged_df.iterrows():
        num += 1
        input_string = clean_string(row[column_name])

        # 提取各项信息
        material = extract_material(input_string)
        furnace = extract_furnace(input_string, material)
        standard = extract_standard(input_string)
        thickness, length, width = extract_dimensions(input_string)

        # 获取 df1 中的 Filename 列
        filename = row['Filename'] if 'Filename' in merged_df.columns else None

        # 从合并后的 df2 中提取 Entry Time、Delivery Time 和 Batch
        entry_time = row['Entry Time'] if 'Entry Time' in merged_df.columns else None
        delivery_time = row['Delivery Time'] if 'Delivery Time' in merged_df.columns else None
        batch = row['Batch'] if 'Batch' in merged_df.columns else None

        # 将提取到的数据添加到结果列表中
        result_data.append({
            "Filename": filename,  # 加入Filename列
            "Material": material,
            "Furnace": furnace,
            "Standard": standard,
            "Thickness": thickness,
            "Width": width,
            "Length": length,
            "Entry Time": entry_time,  # 来自 df2 的数据
            "Delivery Time": delivery_time,  # 来自 df2 的数据
            "Batch": batch  # 来自 df2 的数据
        })

    print(f'num = {num}')
    # 转换为DataFrame
    result_df = pd.DataFrame(result_data)

    # 返回结果的DataFrame
    return result_df