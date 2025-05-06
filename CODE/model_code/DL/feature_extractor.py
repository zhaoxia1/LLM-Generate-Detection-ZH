import torch, random, jieba, json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# 计算统计特征
def compute_stat_features(input_ids, attention_mask):
    """
    计算三个简单的统计特征：
      1. 归一化的序列有效长度 = attention_mask.sum(dim=1) / seq_length
      2. token id 均值（归一化）
      3. token id 标准差（归一化）
    """
    seq_length = input_ids.size(1)
    norm_len = attention_mask.sum(dim=1, keepdim=True).float() / seq_length
    mean_token = input_ids.float().mean(dim=1, keepdim=True) / 50000.
    std_token = input_ids.float().std(dim=1, keepdim=True) / 50000.
    stat_features = torch.cat([norm_len, mean_token, std_token], dim=1)
    return stat_features  # shape: (batch_size, 3)


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, model_encoder=None, source_encoder=None, augment_prob=0.2):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_encoder = model_encoder
        self.source_encoder = source_encoder
        self.augment_prob = augment_prob  # 数据增强的概率

        if model_encoder is None:
            self.model_encoder = LabelEncoder()
            model_labels = [item["model"] for item in data if "model" in item]
            if model_labels:
                self.model_encoder.fit(model_labels)

        if source_encoder is None:
            self.source_encoder = LabelEncoder()
            source_labels = [item["source"] for item in data if "source" in item]
            if source_labels:
                self.source_encoder.fit(source_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        # 数据增强
        if random.random() < self.augment_prob:
            text = self.augment_text(text)

        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                truncation=True)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        try:
            label = torch.tensor(item["label"], dtype=torch.long)
        except KeyError:
            label = torch.tensor(-1, dtype=torch.long)

        ### BEST MODEL MODIFIED: 计算统计特征
        stat_features = compute_stat_features(input_ids.unsqueeze(0), attention_mask.unsqueeze(0)).squeeze(0)

        if "model" in item:
            model_label = torch.tensor(self.model_encoder.transform([item["model"]])[0], dtype=torch.long)
            source_label = torch.tensor(self.source_encoder.transform([item["source"]])[0], dtype=torch.long)
            return input_ids, attention_mask, label, model_label, source_label, stat_features
        else:
            return input_ids, attention_mask, label, stat_features

    def augment_text(self, text):
        """
        对文本进行简单的数据增强，包括随机插入和乱序。
        """
        words = list(jieba.cut(text))
        augmented_words = words.copy()

        # 随机插入
        if random.random() < 0.3:  # 50% 的概率进行随机插入
            insert_word = random.choice(augmented_words)
            insert_position = random.randint(0, len(augmented_words))
            augmented_words.insert(insert_position, insert_word)

        # 乱序
        if random.random() < 0.3:  # 50% 的概率进行乱序
            random.shuffle(augmented_words)

        # 随机删除
        if random.random() < 0.1:  # 50% 的概率进行随机删除
            delete_position = random.randint(0, len(augmented_words) - 1)
            del augmented_words[delete_position]

        # 随机替换
        if random.random() < 0.3:  # 50% 的概率进行随机替换
            replace_position = random.randint(0, len(augmented_words) - 1)
            replace_word = random.choice(augmented_words)
            augmented_words[replace_position] = replace_word
        return ''.join(augmented_words)


# 读取 json 文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data