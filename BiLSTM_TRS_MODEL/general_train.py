"""
    自主构建训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import re
import json
import copy
from tqdm import tqdm


TRAIN_CAR_DATA_DIR = '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/V2_BiLSTM_DATA/train_data/'
TRAIN_CAR_LABEL_DIR = '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/V2_BiLSTM_DATA/train_label/'


TRACK_CLASS_INDEX = 0
TRACK_ID_INDEX = 1
TRACK_X1_INDEX = 2
TRACK_Y1_INDEX = 3
TRACK_X2_INDEX = 4
TRACK_Y2_INDEX = 5
TRACK_CONFIDENCE_INDEX = 6
TRACK_FRAME_COUNT_INDEX = 7
TRACK_FRAME_WIDTH_INDEX = 8
TRACK_FRAME_HEIGHT_INDEX = 9

SEQUENCE_LENGTH = 20

MAX_EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKS = 16


INPUT_SIZE = 5
DROPOUT_RATE = 0.5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.0001
NUM_CLASSES = 2


class BiLSTM_Model(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, numClasses):
        super(BiLSTM_Model, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numClasses = numClasses

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=self.inputSize,
            hidden_size=self.hiddenSize,
            num_layers=self.numLayers,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT_RATE if numLayers > 1 else 0
        )

        # 注意层
        self.attention = nn.Sequential(
            nn.Linear(self.hiddenSize * 2, self.hiddenSize),
            nn.Tanh(),
            nn.Linear(self.hiddenSize, 1)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(self.hiddenSize * 2, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.numClasses)
        )

    def forward(self, x):
        # LSTM输出
        lstm_out, (hidden, cell) = self.lstm(x)

        # 注意力机制
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # 分类
        out = self.classifier(context_vector)
        return out

class CarTrackDataset(Dataset):
    def __init__(self, SEQUENCE_LENGTH):
        self.name = 'Car Track Dataset'
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.specificDataPath = []
        self.specificLabelPath = []
        self.IDCarTrainData = {}
        # 存储训练特征的相关属性
        self.sequenceFeatures = []
        self.sequenceLabels = []
        # 获取具体训练数据
        self.get_train_data()

    def sort_files_by_number(self, fileList):
        """
        按文件名中的数字排序
        """

        def extract_number(filename):
            # 使用正则表达式提取数字
            match = re.search(r'v(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0

        return sorted(fileList, key=extract_number)


    def get_train_data(self):
        dataFileName = self.sort_files_by_number(os.listdir(TRAIN_CAR_DATA_DIR))
        labelFileName = self.sort_files_by_number(os.listdir(TRAIN_CAR_LABEL_DIR))

        for i in range(len(dataFileName)):
            self.specificDataPath.append(os.path.join(TRAIN_CAR_DATA_DIR, dataFileName[i]))
            self.specificLabelPath.append(os.path.join(TRAIN_CAR_LABEL_DIR, labelFileName[i]))

        # load data
        for i in range(len(self.specificDataPath)):
            with open(self.specificDataPath[i], 'r') as f:
                specificData = json.load(f)
            specificFPS = 0
            specificLabel = []
            with open(self.specificLabelPath[i], 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('fps,'):
                        specificFPS = int(line.split(',')[1])
                    else:
                        splitStr = line.split(',')
                        specificLabel.append(copy.deepcopy((splitStr[0].strip(), splitStr[1].strip())))

            changeLabelCount = 0
            # 根据车辆ID整合轨迹点
            for key, value in specificData.items():
                for j in range(len(value)):
                    if f"{value[j][TRACK_ID_INDEX]}" not in self.IDCarTrainData:
                        self.IDCarTrainData[f"{value[j][TRACK_ID_INDEX]}"] = []

                    normalX = (value[j][TRACK_X2_INDEX] - value[j][TRACK_X1_INDEX]) / value[j][TRACK_FRAME_WIDTH_INDEX]
                    normalY = (value[j][TRACK_Y2_INDEX] - value[j][TRACK_Y1_INDEX]) / value[j][TRACK_FRAME_HEIGHT_INDEX]
                    # 计算宽高比
                    widthRate = (value[j][TRACK_X2_INDEX] - value[j][TRACK_X1_INDEX]) / value[j][TRACK_FRAME_WIDTH_INDEX]
                    heightRate = (value[j][TRACK_Y2_INDEX] - value[j][TRACK_Y1_INDEX]) / value[j][TRACK_FRAME_HEIGHT_INDEX]
                    # 设置具体的label
                    labelCount = 0
                    for k in range(len(specificLabel)):
                        if specificLabel[k][0] == f"{value[j][TRACK_ID_INDEX]}":
                            if specificLabel[k][1] == 'accident':
                                if changeLabelCount >= self.SEQUENCE_LENGTH:
                                    labelCount = 1
                                else:
                                    changeLabelCount += 1
                    self.IDCarTrainData[f"{value[j][TRACK_ID_INDEX]}"].append(copy.deepcopy((specificFPS, normalX, normalY, labelCount, widthRate, heightRate)))


        # 整合所有的训练数据信息，便于后续训练获取
        maxLength = 0
        for key, value in self.IDCarTrainData.items():
            if len(value) < self.SEQUENCE_LENGTH:
                continue
            else:
                if len(value) > maxLength:
                    maxLength = len(value)

        for key, value in self.IDCarTrainData.items():
            if len(value) < self.SEQUENCE_LENGTH:
                continue
            else:
                tempFeatures = []
                fps = 0
                normalX = 0
                normalY = 0
                tempLabel = 0
                widthRate = 0
                heightRate = 0
                for i in range(min(len(value), maxLength)):
                    fps, normalX, normalY, tempLabel, widthRate, heightRate = value[i]
                    tempFeatures.append(copy.deepcopy([fps, normalX, normalY, widthRate, heightRate]))


                if len(tempFeatures) < maxLength:
                    for i in range(maxLength - len(tempFeatures)):
                        tempFeatures.append(copy.deepcopy([fps, normalX, normalY, widthRate, heightRate]))


                for i in range(maxLength-self.SEQUENCE_LENGTH):
                    self.sequenceFeatures.append(copy.deepcopy(tempFeatures[i:i+self.SEQUENCE_LENGTH]))
                    self.sequenceLabels.append(copy.deepcopy(tempLabel))

        self.sequenceFeatures = torch.FloatTensor(self.sequenceFeatures)
        self.sequenceLabels = torch.LongTensor(self.sequenceLabels)


    def __len__(self):
        return len(self.sequenceFeatures)

    def __getitem__(self, item):
        return self.sequenceFeatures[item], self.sequenceLabels[item]



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 实例化与加载数据集
    CarTrackDatasetInstance = CarTrackDataset(SEQUENCE_LENGTH)
    # 创建数据加载器
    trainDataLoader = DataLoader(
        CarTrackDatasetInstance,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKS
    )

    BiLSTM_Model_Instance = BiLSTM_Model(inputSize=INPUT_SIZE, hiddenSize=HIDDEN_SIZE,
                         numLayers=NUM_LAYERS, numClasses=NUM_CLASSES)

    BiLSTM_Model_Instance.to(device)
    BiLSTM_Model_Instance.train()


    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(BiLSTM_Model_Instance.parameters(), lr=LEARNING_RATE)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )


    totalTrainLoss = 0
    for epoch in range(MAX_EPOCHS):
        batchCount = 0
        pbar = tqdm(trainDataLoader, desc=f'Epoch {epoch+1}/{MAX_EPOCHS}')
        for inputs, labels in pbar:
            batchCount += 1
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = BiLSTM_Model_Instance(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(BiLSTM_Model_Instance.parameters(), max_norm=1.0)



            optimizer.step()
            optimizer.zero_grad()

            totalTrainLoss += loss.item()

            # 更新进度条
            if batchCount % 100 == 0:
                pbar.set_postfix({'loss': loss.item()})

            # _, predicted = torch.max(outputs.data, 1)
            # print(predicted, labels)

        averageLoss = totalTrainLoss / len(trainDataLoader)
        # 学习率调整
        scheduler.step(averageLoss)
        print(f"current epoch: {epoch+1}, average loss: {averageLoss}")




if __name__ == '__main__':
    main()


