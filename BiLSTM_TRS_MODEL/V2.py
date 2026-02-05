
import torch
import torch.nn as nn
import copy
import math

# 3. BiLSTM+Transformer模型
class noBiLSTMTransformer(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2,
                 num_heads=8, dropout=0.2, num_classes=3):
        super(noBiLSTMTransformer, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.lstm_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, seq_lens=None):
        batch_size, seq_len, _ = x.shape

        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)
        lstm_out = self.dropout(lstm_out)

        lstm_out = self.lstm_fc(lstm_out)
        lstm_out = self.ln2(lstm_out)

        if seq_lens is not None:
            mask = self._create_mask(seq_lens, seq_len).to(x.device)
        else:
            mask = None

        transformer_out = self.transformer_encoder(lstm_out, src_key_padding_mask=mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(transformer_out)
            transformer_out = transformer_out.masked_fill(mask_expanded, 0)
            valid_counts = (~mask).float().sum(dim=1, keepdim=True)
            pooled = transformer_out.sum(dim=1) / valid_counts.clamp(min=1)
        else:
            pooled = transformer_out.mean(dim=1)

        output = self.fc(pooled)

        return output

    def _create_mask(self, seq_lens, max_len):
        batch_size = len(seq_lens)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(seq_lens):
            if length < max_len:
                mask[i, length:] = False
        return mask



class BiLSTMTransformer:
    def __init__(self):
        self.name = 'BiLSTMTransformer'
        self.historyBasicInfo = {}
        self.basedFrameCountInfo = {}
        self.currentVideoWidth = 0
        self.currentVideoHeight = 0
        self.displacementIntervalFrameCount = 5

        # 超速行为相关参数
        self.rashRelatedParaOne = 0.003
        self.rashRelatedParaTwo = 8


        # 暂存信息的属性
        self.tempRecordDisplacement = {}
        self.tempRecordDispSpeed = {}
        self.tempRecordRashTwoCount = {}
        self.tempRecordRashStatusList = []

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
        self.historyBasicInfo[basicInfo[0]].append(copy.deepcopy(basicInfo))
        # 处理与清晰数据等
        if f"{basicInfo[-3]}" not in self.basedFrameCountInfo:
            self.basedFrameCountInfo[f"{basicInfo[-3]}"] = []
        self.basedFrameCountInfo[f"{basicInfo[-3]}"].append(copy.deepcopy((basicInfo[2], basicInfo[3], basicInfo[4], basicInfo[5], basicInfo[6], basicInfo[7])))

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

            print(self.tempRecordRashStatusList)

            # 接下来取出候选超速列表中的id进行进一步确认处理

