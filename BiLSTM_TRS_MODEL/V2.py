
import torch
import torch.nn as nn
import copy


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

    def get_basic_info(self, basicInfo):
        if basicInfo[0] not in self.historyBasicInfo:
            self.historyBasicInfo[basicInfo[0]] = []
            # 换摄像头了
            self.basedFrameCountInfo = {}
            self.currentVideoWidth = basicInfo[-2]
            self.currentVideoHeight = basicInfo[-1]
        self.historyBasicInfo[basicInfo[0]].append(copy.deepcopy(basicInfo))
        # 处理与清晰数据等
        if f"{basicInfo[-3]}" not in self.basedFrameCountInfo:
            self.basedFrameCountInfo[f"{basicInfo[-3]}"] = []
        self.basedFrameCountInfo[f"{basicInfo[-3]}"].append(copy.deepcopy((basicInfo[2], basicInfo[3], basicInfo[4], basicInfo[5], basicInfo[6], basicInfo[7])))

    # 计算目标基于像素的特定帧数的过去的位移
    def calculate_frame_pixel_displacement(self):
        pass

    # 数据预处理与准备
    def data_preprocess_preparation(self):
        # 计算目标基于像素的特定帧数的过去的位移
        self.calculate_frame_pixel_displacement()

