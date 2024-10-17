import os
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from backend import utils as ut
from backend import constants

class ImgRec:
    def __init__(self):
        self.reader = None
        self.allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()-/* '
        self.average_confidences = []
        self.Batch = 0
        self.current_df = None

    def set_reader(self, reader_instance):
        self.reader = reader_instance

    def clear_confidences(self):
        # 清空置信度列表
        self.average_confidences = []

    # 进行OCR识别
    def Rec_fun(self, image, file_name, correct_text=None):
        '''
        传入：图像，图像文件名，图像输出路径，正确编码
        返回：图像文件名，识别编码，识别完成时间，平均识别准确度，准确率
        '''
        # 如果图像是PIL对象，转换为numpy数组
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 文字识别
        results = self.reader.readtext(image, allowlist=self.allowlist, link_threshold=0.8, paragraph=False)

        # 提取识别结果
        recognition_text = ''
        total_confidence = 0.0
        for (bbox, text, prob) in results:
            recognition_text += text + ' '
            total_confidence += prob

        recognition_text = ut.process_steel_code(recognition_text)
        average_confidence = total_confidence / len(results) if results else 0.0

        accuracy = 0
        if correct_text is not None:
            # 计算准确率
            accuracy = ut.calculate_accuracy(recognition_text, correct_text) if correct_text else None
        accuracy = "{:.2%}".format(accuracy)

        # 保存处理完成时间
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 返回附带准确率的结果
        return file_name, recognition_text, average_confidence, accuracy, timestamp
    #文件夹识别
    def process_images_from_folder(self, folder_path, progress_placeholder):
        """处理文件夹中的图像并进行 OCR 识别"""
        data1, data2 = [], []

        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.bmp'))]
        total_images = len(image_files)
        if total_images==0:
            return 0
        # 加载table.csv
        label_path = os.path.join(folder_path, "label.csv")
        label_data = None
        if os.path.exists(label_path):
            label_data = pd.read_csv(label_path)  # 确保加载正确

        self.clear_confidences()
        self.Batch += 1

        # 获取当前时间和交付时间，以及批次
        entry_time = datetime.now().strftime('%Y-%m-%d')  # 当前日期
        delivery_time = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')  # 30天后的日期
        Batch = 'Q' + str(self.Batch)

        for idx, file_name in enumerate(image_files):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            #获取正确编码
            correct_text = None
            if label_data is not None:
                # 查找该图像对应的正确编码
                correct_text = label_data.loc[label_data['Filename'] == file_name, 'Recognized Text'].values[0]
            # 识别图像并计算准确率
            file_name, recognition_text, average_confidence, accuracy, timestamp = self.Rec_fun(image, file_name, correct_text)
            self.average_confidences.append(average_confidence)
            # 将数据追加到data中
            data1.append(
                {"Filename": file_name, "Recognized Text": recognition_text, "Average Confidence": average_confidence,
                 "Accuracy": accuracy, "Timestamp": timestamp})
            data2.append(
                {"Filename": file_name, "Timestamp": timestamp,
                 "Entry Time": entry_time, "Delivery Time": delivery_time, "Batch": Batch})
            # 更新进度条
            progress_placeholder.progress((idx + 1) / total_images)

        if data1:
            ut.append_to_csv(data1, constants.CSV_FILE_PATH)
            df = pd.DataFrame(data1)
            self.current_df = df
        if data2:
            ut.append_to_csv(data2, constants.CSV_OTHER_PATH)

        return total_images
    #上传文件识别
    def process_uploaded_images(self, uploaded_files, progress_placeholder):
        """处理上传的图片并返回识别结果"""
        data1, data2 = [], []
        total_files = len(uploaded_files)  # 上传文件总数

        if total_files == 0:
            return  0  # 如果还为上传图片，返回None

        # 遍历每个上传的文件
        self.clear_confidences()
        self.Batch += 1

        # 获取当前时间和交付时间
        entry_time = datetime.now().strftime('%Y-%m-%d')  # 当前日期
        delivery_time = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')  # 30天后的日期
        Batch = 'Q' + str(self.Batch)

        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)  # 打开图片
            # 识别图像并计算准确率
            file_name, recognition_text, average_confidence, accuracy, timestamp = self.Rec_fun(image, uploaded_file.name)
            self.average_confidences.append(average_confidence)

            # 将数据追加到data中
            data1.append(
                {"Filename": file_name, "Recognized Text": recognition_text, "Average Confidence": average_confidence,
                 "Accuracy": accuracy, "Timestamp": timestamp})
            data2.append(
                {"Filename": file_name, "Timestamp": timestamp,
                 "Entry Time": entry_time, "Delivery Time": delivery_time, "Batch": Batch})

            # 更新进度条
            progress_placeholder.progress((idx + 1) / total_files)

        if data1:
            ut.append_to_csv(data1, constants.CSV_FILE_PATH)
            df = pd.DataFrame(data1)
            self.current_df = df
        if data2:
            ut.append_to_csv(data2, constants.CSV_OTHER_PATH)

        return total_files
    #视频取帧识别
    def process_video_from_folder(self, frame_interval, video_path, progress_placeholder):

        ut.extract_unique_frames_from_video(frame_interval, video_path) #取帧，去重
        self.process_images_from_folder(constants.frames_final_folder, progress_placeholder)


