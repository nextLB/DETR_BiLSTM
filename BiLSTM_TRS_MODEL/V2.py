
import sys

import cv2
import torch

sys.path.append('/home/next_lb/桌面/next/DETR_BiLSTM/BiLSTM_TRS_MODEL')
from general_train import *
import copy
import math




class BiLSTMTransformer:
    def __init__(self):
        self.name = 'BiLSTMTransformer'
        self.fps = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.historyBasicInfo = {}
        self.basedFrameCountInfo = {}
        self.currentVideoWidth = 0
        self.currentVideoHeight = 0
        self.displacementIntervalFrameCount = 5

        # 超速行为相关参数
        self.rashRelatedParaOne = 0.0012
        self.rashRelatedParaTwo = 10

        # 是否出现交通事故的相关参数
        self.accidentRelatedParaOne = 0.03
        self.accidentRelatedParaTwo = 10


        # 暂存信息的属性
        self.tempRecordDisplacement = {}
        self.tempRecordDispSpeed = {}
        self.tempRecordRashTwoCount = {}
        self.tempRecordRashStatusList = []
        self.tempRecordAccidentCount = {}

        # 加载Bi LSTM模型
        self.BiLSTMModel = BiLSTM_Model(
            inputSize=INPUT_SIZE,
            hiddenSize=HIDDEN_SIZE,
            numLayers=NUM_LAYERS,
            numClasses=NUM_CLASSES
        )
        # 加载模型权重
        self.BiLSTMModel.load_state_dict(torch.load(f"{SAVE_MODEL_PATH_DIR}/best_Bi_LSTM_model.pth", map_location=self.device))
        self.BiLSTMModel.to(self.device)

        self.IDTrackInfo = {}


    def get_nth_last_from_dict(self, dictionary, n):
        """
        获取字典倒数第n个元素

        参数:
            dictionary: 字典
            n: 倒数第n个（1表示最后一个，2表示倒数第二个...）

        返回:
            (key, value) 元组，如果不存在返回None
        """
        if not dictionary or len(dictionary) < n:
            return None

        # 将字典项转换为列表
        items_list = list(dictionary.items())
        # 获取倒数第n个
        return items_list[-n]

    def euclidean_distance(self, point1, point2):
        """计算两个点之间的欧几里得距离。

        参数:
        point1: tuple, 第一个点的坐标 (x1, y1)
        point2: tuple, 第二个点的坐标 (x2, y2)

        返回:
        float, 欧几里得距离

        """
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def get_basic_info(self, basicInfo):

        if basicInfo[0] not in self.historyBasicInfo:
            self.historyBasicInfo[basicInfo[0]] = []
            # 换摄像头了
            self.basedFrameCountInfo = {}
            self.currentVideoWidth = basicInfo[-2]
            self.currentVideoHeight = basicInfo[-1]
            self.tempRecordRashTwoCount = {}
            self.tempRecordAccidentCount = {}
            self.IDTrackInfo = {}
            cap = cv2.VideoCapture(f"/home/next_lb/桌面/next/DETR_BiLSTM/test_videos/{basicInfo[0]}")
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        self.historyBasicInfo[basicInfo[0]].append(copy.deepcopy(basicInfo))
        # 处理与清晰数据等
        if f"{basicInfo[-3]}" not in self.basedFrameCountInfo:
            self.basedFrameCountInfo[f"{basicInfo[-3]}"] = []
        self.basedFrameCountInfo[f"{basicInfo[-3]}"].append(copy.deepcopy((basicInfo[2], basicInfo[3], basicInfo[4], basicInfo[5], basicInfo[6], basicInfo[7])))

        # 处理出一个按照追踪ID存储的历史数据信息
        if basicInfo[0] in self.historyBasicInfo:
            if f"{basicInfo[2]}" not in self.IDTrackInfo:
                self.IDTrackInfo[f"{basicInfo[2]}"] = []
            # 处理成可用的feature再加入到缓存中
            normalX = (basicInfo[3] + basicInfo[5]) / (2 * basicInfo[9])
            normalY = (basicInfo[4] + basicInfo[6]) / (2 * basicInfo[10])
            # 计算宽高比
            widthRate = (basicInfo[5] - basicInfo[3]) / basicInfo[9]
            heightRate = (basicInfo[6] - basicInfo[4]) / basicInfo[10]
            self.IDTrackInfo[f"{basicInfo[2]}"].append(copy.deepcopy((self.fps, normalX, normalY, widthRate, heightRate)))


    # 计算目标基于像素的特定帧数的过去的位移
    def calculate_frame_pixel_displacement(self):
        if self.basedFrameCountInfo and len(self.basedFrameCountInfo) >= self.displacementIntervalFrameCount:
            theLastOneData = self.get_nth_last_from_dict(self.basedFrameCountInfo, 1)
            intervalData = self.get_nth_last_from_dict(self.basedFrameCountInfo, self.displacementIntervalFrameCount)
            # 计算位移差值
            for i in range(len(theLastOneData[1])):
                # 查找相同ID,并计算位移
                for j in range(len(intervalData[1])):
                    if theLastOneData[1][i][0] == intervalData[1][j][0]:
                        centerPointOne = ((theLastOneData[1][i][3] - theLastOneData[1][i][1])/self.currentVideoWidth, (theLastOneData[1][i][4] - theLastOneData[1][i][2])/self.currentVideoHeight)
                        centerPointTwo = ((intervalData[1][i][3] - intervalData[1][i][1])/self.currentVideoWidth, (intervalData[1][i][4] - intervalData[1][i][2])/self.currentVideoHeight)
                        displacementValue = self.euclidean_distance(centerPointOne, centerPointTwo)
                        self.tempRecordDisplacement[f"{theLastOneData[1][i][0]}"] = displacementValue


    # 计算目标基于像素位移当前的大致速率
    def calculate_current_displacement_speed(self):
        if self.tempRecordDisplacement:
            for key, value in self.tempRecordDisplacement.items():
                self.tempRecordDispSpeed[key] = value / self.displacementIntervalFrameCount


    # 数据预处理与准备
    def data_preprocess_preparation(self):
        # 重置暂存信息相关属性
        self.tempRecordDisplacement = {}
        self.tempRecordDispSpeed = {}
        self.tempRecordRashStatusList = []


        # 计算目标基于像素的特定帧数的过去的位移
        self.calculate_frame_pixel_displacement()

        # 计算目标基于像素位移当前的大致速率
        self.calculate_current_displacement_speed()

    # 进行行为的判别
    def conduct_behavior_assessment(self):
        if self.tempRecordDispSpeed:
            for key, value in self.tempRecordDispSpeed.items():
                if value > self.rashRelatedParaOne:
                    if f"{key}" not in self.tempRecordRashTwoCount:
                        self.tempRecordRashTwoCount[f"{key}"] = 0
                    self.tempRecordRashTwoCount[f"{key}"] += 1
            # 挑选第一轮的超速候选ID
            for key, value in self.tempRecordRashTwoCount.items():
                if value > self.rashRelatedParaTwo:
                    self.tempRecordRashStatusList.append(key)

            # TODO: 对于一些超速车辆还要进行进一步的判定的

            # 先打印一下是否有超速危险的检测
            for i in range(len(self.tempRecordRashStatusList)):
                print(f'基于简单的规则--------------追踪ID为: {self.tempRecordRashStatusList[i]}的车辆有超速的危险')
            print('-'*30)
            print('-'*30)
            print('-'*30)


            # 接下来取出候选超速列表中的id进行进一步确认处理关于事故的检测
            for i in range(len(self.tempRecordRashStatusList)):
                for j in range(i+1, len(self.tempRecordRashStatusList)):
                    # 查找这两个ID对应的识别信息
                    theLastOneData = self.get_nth_last_from_dict(self.basedFrameCountInfo, 1)
                    iInfo = None
                    jInfo = None
                    for k in range(len(theLastOneData[1])):
                        if iInfo != None and jInfo != None:
                            break
                        else:
                            if theLastOneData[1][k][0] == int(self.tempRecordRashStatusList[i]):
                                iInfo = theLastOneData[1][k]
                            elif theLastOneData[1][k][0] == int(self.tempRecordRashStatusList[j]):
                                jInfo = theLastOneData[1][k]

                    if iInfo and jInfo:
                        # 计算两个追踪ID车辆之间归一化后的中心坐标点之间的距离
                        x1NormalDiff = (iInfo[3] - iInfo[1]) / self.currentVideoWidth
                        y1NormalDiff = (iInfo[4] - iInfo[2]) / self.currentVideoHeight
                        x2NormalDiff = (jInfo[3] - jInfo[1]) / self.currentVideoWidth
                        y2NormalDiff = (jInfo[4] - jInfo[2]) / self.currentVideoHeight
                        normalDistance = self.euclidean_distance((x1NormalDiff, y1NormalDiff), (x2NormalDiff, y2NormalDiff))
                        if normalDistance < self.accidentRelatedParaOne:
                            if f"{self.tempRecordRashStatusList[i]}_{self.tempRecordRashStatusList[j]}" not in self.tempRecordAccidentCount:
                                self.tempRecordAccidentCount[f"{self.tempRecordRashStatusList[i]}_{self.tempRecordRashStatusList[j]}"] = 0
                            self.tempRecordAccidentCount[f"{self.tempRecordRashStatusList[i]}_{self.tempRecordRashStatusList[j]}"] += 1

            for key, value in self.tempRecordAccidentCount.items():
                print(f"基于简单的规则-------------- {key} 发生了车祸似乎")



    # 使用机器学习等方式进行行为的判定
    def determine_behavior_machine_learning(self):
        for key, value in self.IDTrackInfo.items():
            if len(value) >= SEQUENCE_LENGTH:
                inputFeatures = value[-SEQUENCE_LENGTH:]
                inputFeatures = torch.FloatTensor(inputFeatures).unsqueeze(0).to(self.device)
                # 预测
                with torch.no_grad():
                    outputs = self.BiLSTMModel(inputFeatures)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probabilities, dim=1)
                # 提取tensor中的数值
                predictionValue = prediction.item()  # 提取预测类别（0或1）
                confidenceValue = confidence.item()  # 提取置信度（浮点数）
                print(f"基于神经网络模型的方式--------------------------- track id 为: {key}, 当前BiLSTM模型预测的行为是: {predictionValue}, 置信度是: {confidenceValue:.4f}")





