# model.py - 修正版
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import math

from config import Config


class PositionEmbeddingSine(nn.Module):
    """正弦位置编码"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

    def forward(self, x):
        # x: [batch, channel, height, width]
        not_mask = torch.ones(x.shape[-2:], device=x.device)
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        return pos.unsqueeze(0)  # [1, C, H, W]


class TransformerEncoderLayer(nn.Module):
    """自定义Transformer编码器层，支持位置编码"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, pos):
        # src: [S, B, C], pos: [S, B, C]
        q = k = src + pos
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    """自定义Transformer解码器层"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # tgt: [T, B, C], memory: [S, B, C]
        # pos: [S, B, C], query_pos: [T, B, C]

        # 自注意力
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 交叉注意力
        q = tgt + query_pos
        k = memory + pos
        tgt2 = self.multihead_attn(q, k, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        return output


class TransformerDecoder(nn.Module):
    """Transformer解码器"""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
        return output


class MLP(nn.Module):
    """简单的多层感知机"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """DETR模型实现"""

    def __init__(self):
        super().__init__()

        # 骨干网络（ResNet50）
        backbone = resnet50(pretrained=True)
        # 移除最后的全连接层和池化层
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # 转换层
        self.conv = nn.Conv2d(2048, Config.HIDDEN_DIM, 1)

        # 位置编码
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=Config.HIDDEN_DIM // 2,
            normalize=True
        )

        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=Config.HIDDEN_DIM,
            nhead=Config.NHEADS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT
        )
        self.encoder = TransformerEncoder(encoder_layer, Config.NUM_ENCODER_LAYERS)

        # Transformer解码器
        decoder_layer = TransformerDecoderLayer(
            d_model=Config.HIDDEN_DIM,
            nhead=Config.NHEADS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT
        )
        self.decoder = TransformerDecoder(decoder_layer, Config.NUM_DECODER_LAYERS)

        # 查询向量（object queries）
        self.query_embed = nn.Embedding(Config.NUM_QUERIES, Config.HIDDEN_DIM)

        # 预测头
        self.bbox_embed = MLP(Config.HIDDEN_DIM, Config.HIDDEN_DIM, 4, 3)
        self.class_embed = nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES + 1)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

        # 初始化预测头
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias.data, bias_value)

    def forward(self, x):
        # 骨干网络提取特征
        features = self.backbone(x)

        # 调整通道数
        features = self.conv(features)

        # 添加位置编码
        pos_embed = self.position_embedding(features)

        # 展平特征图 (B, C, H, W) -> (H*W, B, C)
        batch_size, _, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (H*W, B, C)

        # 编码器
        memory = self.encoder(features, pos=pos_embed)

        # 解码器查询
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # 解码器
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(
            tgt, memory,
            pos=pos_embed,
            query_pos=query_embed
        )

        # 预测
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}


class PostProcess(nn.Module):
    """后处理：将模型输出转换为检测结果"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 转换为绝对坐标
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = out_bbox * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            # 过滤低置信度检测
            keep = s > 0.7
            results.append({
                'scores': s[keep],
                'labels': l[keep],
                'boxes': b[keep]
            })

        return results