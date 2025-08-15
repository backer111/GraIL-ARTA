import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为Transformer提供序列位置信息
    """
    def __init__(self, d_model, max_len=100, device='cpu'):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # 确保pe在正确的设备上
        self.device = device
        self.to(device)

    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model]
        """
        # 确保pe和x在同一设备上
        return x + self.pe[:, :x.size(1), :].to(x.device)


class TransformerAgent(nn.Module):
    """
    基于Transformer的Agent，处理规则嵌入序列并生成调整向量
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4, 
                 dropout=0.1, max_seq_len=50, no_rule_idx=0, device='cpu'):
        super(TransformerAgent, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.no_rule_idx = no_rule_idx
        self.device = device
        
        # 位置编码，传递设备参数
        self.pos_encoder = PositionalEncoding(input_dim, max_seq_len, device)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 用于辅助任务的规则预测层
        self.rule_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 每条规则的得分
        )
        
        # 规则嵌入矩阵，将在外部注册
        self.rule_embeddings = None
        
        # 初始化参数
        self._init_parameters()
        
        # 移动到指定设备
        self.to(device)
        
    def _init_parameters(self):
        """
        初始化模型参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def register_rule_embeddings(self, embeddings):
        """
        注册规则嵌入矩阵
        
        参数:
            embeddings: 规则嵌入矩阵，形状为 [num_rules, embedding_dim]
        """
        # 确保嵌入也在正确的设备上
        self.rule_embeddings = nn.Parameter(embeddings.to(self.device), requires_grad=False)
        
    def forward(self, rule_ids, rule_masks=None):
        """
        前向传播
        
        参数:
            rule_ids: 规则ID，形状为 [batch_size, max_rules]
            rule_masks: 规则掩码，形状为 [batch_size, max_rules]，指示哪些位置有有效的规则
            
        返回:
            adjustment: 调整向量，形状为 [batch_size, output_dim]
            rule_scores: 规则得分，形状为 [batch_size, max_rules]，用于辅助任务
        """
        batch_size, max_rules = rule_ids.shape
        
        # 获取规则嵌入
        if self.rule_embeddings is None:
            raise ValueError("规则嵌入矩阵未注册，请先调用register_rule_embeddings方法")
        
        # 将规则ID转换为嵌入向量 [batch_size, max_rules, input_dim]
        rule_embeds = self.rule_embeddings[rule_ids]
        
        # 添加位置编码
        rule_embeds = self.pos_encoder(rule_embeds)
        
        # 创建注意力掩码（如果提供了rule_masks）
        if rule_masks is not None:
            # 创建一个布尔掩码，True表示该位置应该被掩盖
            # Transformer中，被掩盖的位置设置为-inf
            attn_mask = ~rule_masks
            
            # 对于全都是False的行（即全部是no_rule_idx的情况），不应用掩码
            all_masked_rows = attn_mask.all(dim=1)
            attn_mask[all_masked_rows] = False
        else:
            attn_mask = None
        
        # 通过Transformer编码器
        # 在PyTorch中，如果key_padding_mask为True，则对应位置会被mask掉
        transformer_out = self.transformer_encoder(
            rule_embeds, 
            src_key_padding_mask=attn_mask if attn_mask is not None else None
        )
        
        # 计算规则得分（用于辅助任务）
        rule_scores = self.rule_predictor(transformer_out).squeeze(-1)
        
        # 如果有掩码，对无效规则的得分设为一个很小的值
        if rule_masks is not None:
            rule_scores = rule_scores.masked_fill(~rule_masks, -1e9)
        
        # 对序列进行注意力池化得到最终表示
        # 使用softmax来计算每个规则的权重
        attn_weights = F.softmax(rule_scores, dim=1).unsqueeze(2)  # [batch_size, max_rules, 1]
        pooled = torch.sum(transformer_out * attn_weights, dim=1)  # [batch_size, input_dim]
        
        # 生成调整向量
        adjustment = self.output_layer(pooled)
        
        return adjustment, rule_scores
    
    def predict_adjustment(self, rule_ids, rule_masks=None):
        """
        只预测调整向量，不返回规则得分
        """
        adjustment, _ = self.forward(rule_ids, rule_masks)
        return adjustment
    
    def compute_aux_loss(self, rule_scores, rule_masks):
        """
        计算辅助损失（规则预测任务）
        
        参数:
            rule_scores: 规则得分，形状为 [batch_size, max_rules]
            rule_masks: 规则掩码，形状为 [batch_size, max_rules]
            
        返回:
            aux_loss: 辅助损失
        """
        # 对于有效的规则，鼓励分配更高的分数
        # 这里使用简单的二元交叉熵损失
        valid_rule_scores = rule_scores.masked_select(rule_masks)
        valid_rule_targets = torch.ones_like(valid_rule_scores)
        
        if len(valid_rule_scores) > 0:
            aux_loss = F.binary_cross_entropy_with_logits(
                valid_rule_scores, 
                valid_rule_targets
            )
        else:
            aux_loss = torch.tensor(0.0, device=self.device)
        
        return aux_loss
    
    def compute_reg_loss(self):
        """
        计算正则化损失，避免过拟合
        
        返回:
            reg_loss: 正则化损失
        """
        # L2正则化
        reg_loss = sum(torch.sum(p ** 2) for p in self.parameters() if p.requires_grad)
        return reg_loss 