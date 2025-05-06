import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class EnhancedMMOE(nn.Module):
    def __init__(self, bert_path, num_experts=3, expert_dim=768,
                 num_labels_2=4, num_labels_3=3, stat_feature_dim=3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BertModel.from_pretrained(bert_path).to(self.device)

        # 冻结和解冻层
        for param in self.bert.parameters():
            param.requires_grad = False
        total_layers = self.bert.config.num_hidden_layers
        for layer_idx in [total_layers -2 ,total_layers - 1]:
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = True
        

        self.gate_networks = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Linear(256, num_experts)
            ).to(self.device) for task in ["label", "model", "source"]
        })
        class ExpertBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.net = nn.Sequential(
                    nn.Linear(dim, dim*2),
                    nn.GELU(),
                    nn.LayerNorm(dim*2),
                    nn.Linear(dim*2, dim),
                    nn.Dropout(0.1)
                ).to(self.device)
                self.ln = nn.LayerNorm(dim).to(self.device)
            def forward(self, x):
                return self.ln(x + self.net(x))
            
        # 共享专家层（增加Dropout）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                ExpertBlock(self.bert.config.hidden_size),
                nn.Dropout(0.1),
                ExpertBlock(self.bert.config.hidden_size)
            ).to(self.device)
            for _ in range(num_experts)
        ]).to(self.device)


        # 任务专家层
        self.task_experts = nn.ModuleDict({
            "label": nn.ModuleList([nn.Linear(expert_dim, expert_dim).to(self.device)
                                    for _ in range(num_experts)]),
            "model": nn.ModuleList([nn.Linear(expert_dim, expert_dim).to(self.device)
                                    for _ in range(num_experts)]),
            "source": nn.ModuleList([nn.Linear(expert_dim, expert_dim).to(self.device)
                                     for _ in range(num_experts)])
        })

        # 任务特征加权（使用可学习参数和softmax）
        self.task_weighting = nn.Linear(expert_dim * 3, 3).to(self.device)

        # 统计特征转换层
        self.stat_feature_transform = nn.Linear(stat_feature_dim, expert_dim).to(self.device)

        # 预测头
        self.label_head = nn.Sequential(
            nn.Linear(expert_dim*2, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 2)
        ).to(self.device)
        self.model_head = nn.Linear(expert_dim, num_labels_2).to(self.device)
        self.source_head = nn.Linear(expert_dim, num_labels_3).to(self.device)

    def forward(self, input_ids, attention_mask, stat_features=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.bert(input_ids, attention_mask)
        x = outputs.last_hidden_state[:, 0, :]

        # 共享专家层处理
        shared_outputs = torch.stack([expert(x) for expert in self.shared_experts], dim=1)
        x = shared_outputs.sum(dim=1)

        # 任务门控
        gate_weights = {task: F.softmax(gate(x), dim=1).unsqueeze(-1)
                        for task, gate in self.gate_networks.items()}

        # 任务专家处理
        task_features = {}
        for task in ["label", "model", "source"]:
            expert_outputs = torch.stack([expert(x) for expert in self.task_experts[task]], dim=1)
            task_features[task] = (gate_weights[task] * expert_outputs).sum(dim=1)

        # 任务特征加权
        combined_features = torch.cat([task_features["label"], task_features["model"], task_features["source"]], dim=-1)
        weights = F.softmax(self.task_weighting(combined_features), dim=-1)
        label_feature = (
                weights[:, 0].unsqueeze(-1) * task_features["label"] +
                weights[:, 1].unsqueeze(-1) * task_features["model"] +
                weights[:, 2].unsqueeze(-1) * task_features["source"]
        )

        # 如果传入了统计特征，则进行转换并拼接到 label_feature 上
        if stat_features is not None:
            stat_features = stat_features.to(self.device)
            stat_transformed = F.relu(self.stat_feature_transform(stat_features))
            label_feature = torch.cat([label_feature, stat_transformed], dim=-1)

        # 预测输出
        label_out = self.label_head(label_feature)
        model_out = self.model_head(task_features["model"])
        source_out = self.source_head(task_features["source"])

        return label_out, model_out, source_out
