# 基于机器学习与深度学习的中文文本分类系统

## 项目描述
本项目聚焦于中文文本的分类预测，融合了机器学习（ML）和深度学习（DL）两种技术。机器学习部分通过提取中文文本的统计、句法、语义及信息论特征来完成预测任务；深度学习部分则采用了EnhancedMMOE模型。

## 环境要求
### 依赖库
```python
torch=2.5.1+cu121
transformers=4.28.1
scikit-learn=1.5.1
pandas=2.2.2
numpy=1.26.4
tqdm=4.66.5
jieba=0.42.1
snownlp=0.12.3
lightgbm=4.6.0
catboost=1.2.8
xgboost=2.1.2
joblib=1.4.2
```
### 环境搭建
```bash
pip install -r requirements.txt
```

### 目录结构
```
project/
├── cn-macbert/            # 预训练模型目录（需包含必要文件）
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── data/                  # 数据目录（需包含训练、验证和测试数据）
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   └── ...
├── model_save/            # 模型保存目录
│   ├── ...
├── pred_data/             # 预测结果保存目录
│   ├── ...
├── model_code/            # 模型代码目录
│   ├── ML/
│   │   ├── features_extractor.py
│   │   ├── model_predict.py
│   ├── DL/
│   │   ├── feature_extractor.py
│   │   ├── model.py
│   │   ├── train_eval.py
│   │   ├── main.py
│   ├── processdata/
│   │   ├── data_process.py
│   └── MODELVOTING.py/
```

## 数据准备
### 1. 数据格式
确保 `data` 文件夹中的数据格式正确，其中 `train.json`、`dev.json` 和 `test.json` 等文件需包含必要的字段，以满足模型训练和预测的需求。

### 2. 预训练模型
确保 `cn-macbert` 文件夹存放了预训练模型相关文件，这些文件是深度学习模型运行的基础。

## 模型训练与预测
### 机器学习模型（ML部分）
#### 1. 特征提取与模型训练
##### 1.1 修改参数
打开 `submit_files/model_code/ML/features_extractor.py` 文件，根据需要修改以下参数：
```python
train_sample_size = 6000  # 采样大小
dev_sample_size = 1000  # 采样大小
feature_num = 15  # 选择特征数量
load_saved_features = False  # 是否加载保存的特征, 运行一次后可将此处改为True
use_svd = False  # 是否降维
use_tfidf = False  # 是否使用TF-IDF特征
```
##### 1.2 执行训练
在终端中执行以下命令：
```bash
python submit_files/model_code/ML/features_extractor.py
```
该脚本将完成以下操作：
- 从 `data` 文件夹中读取训练和验证数据。
- 提取中文文本的特征。
- 选择特征并进行降维（如果需要）。
- 训练多个基模型（SGDClassifier、NaiveBayes、LightGBM、CatBoost、XGBoost）。
- 保存训练好的模型到 `model_save` 文件夹。

#### 2. 模型预测
##### 2.1 测试集预测
在终端中执行以下命令：
```bash
python submit_files/model_code/ML/model_predict.py
```
该脚本将完成以下操作：
- 加载预处理对象和模型。
- 从 `data` 文件夹中读取测试数据。
- 执行集成模型投票预测和单模型独立预测。
- 将预测结果保存到 `pred_data` 文件夹。

##### 2.2 验证集评估
若想评估模型在验证集上的性能，在 `model_predict.py` 文件中取消注释 `predict_dev()` 函数的调用，然后运行脚本：
```python
if __name__ == '__main__':
    predict_dev()
    # predict_test()
```
该函数将输出集成模型和各单模型在验证集上的准确率、精确率、召回率和F1值。

### 深度学习模型（DL部分）
#### 1. 定义模型
`submit_files/model_code/DL/model.py` 文件中定义了 `EnhancedMMOE` 模型，需确保该文件中的模型定义正确。

#### 2. 使用模型进行预测
以下是加载预训练的BERT模型和 `EnhancedMMOE` 模型并进行预测的示例代码：
```python
import torch
from submit_files.model_code.DL.model import EnhancedMMOE

# 初始化模型
bert_path = 'cn-macbert'
model = EnhancedMMOE(bert_path)

# 加载模型参数（假设已经保存）
model.load_state_dict(torch.load('path/to/model.pth'))
model.eval()

# 准备输入数据
input_ids = torch.randint(0, 1000, (1, 128))
attention_mask = torch.ones((1, 128))
stat_features = torch.randn(1, 3)

# 进行预测
with torch.no_grad():
    label_out, model_out, source_out = model(input_ids, attention_mask, stat_features)
    print("Label output:", label_out)
    print("Model output:", model_out)
    print("Source output:", source_out)
```

### 保存预测结果到JSON文件
若想将预测结果保存到JSON文件，可使用 `save2json` 函数：
```python
from submit_files.model_code.MODELVOTING import save2json

txt_path = 'pred_data/vote_pred.txt'
json_path = 'data/test.json'
save_path = 'data/test_with_labels.json'

save2json(txt_path, json_path, save_path)
```
该函数将读取 `txt_path` 中的标签，将其添加到 `json_path` 的JSON数据中，并保存到 `save_path`。

## 注意事项
- 确保 `data` 文件夹中的数据格式正确，并且文件路径与代码中的一致。
- 如果使用GPU进行训练和预测，确保你的环境已经正确配置了CUDA。
- 在运行代码之前，建议检查各文件中的参数设置是否符合你的需求。