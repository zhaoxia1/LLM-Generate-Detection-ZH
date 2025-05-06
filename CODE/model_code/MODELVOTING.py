import torch
import joblib, os, json
import numpy as np
from torch.nn.functional import softmax
from sklearn.base import ClassifierMixin
from collections import Counter
from DL.model import EnhancedMMOE
from DL.feature_extractor import CustomDataset, read_json
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset 

def single_pred(model, val_loader, f=0, e=0, pred_save_path = None, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluation'):
            if len(batch) == 6:
                input_ids, attention_mask, label, model_label, source_label, stat_features = [x.to(device) for x in batch]
            elif len(batch) == 4:
                input_ids, attention_mask, label, stat_features = [x.to(device) for x in batch]
                model_label = None
                source_label = None
            label_pred, _, _ = model(input_ids, attention_mask, stat_features)
            preds = torch.argmax(label_pred, dim=1).cpu().numpy()
            all_labels.extend(label.tolist())
            all_preds.extend(preds.tolist())

    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as fw:
            for l, p in zip(all_labels, all_preds):
                fw.write(f"{l}\t{p}\n")

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print(f'Epoch {e + 1}, Fold {f + 1}, Val F1: {f1}, Val Acc: {acc}')
    return f1, acc

def vote_dev():
    root_path = os.path.dirname(os.path.dirname(__file__))
    model_path = [os.path.join(root_path,'model_save\best_model_fold_{}.pth'.format(i)) for i in range(1,6) if i != 3]
    tokenizer = BertTokenizer.from_pretrained(os.path.join(root_path,'cn-macbert'))

    val_data = read_json(os.path.join(root_path,'data\dev_dealed.json'))
    val_dataset = CustomDataset(val_data, tokenizer, model_encoder=True, source_encoder=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for i, p in enumerate(model_path):
        model = EnhancedMMOE(bert_path=os.path.join(root_path,'cn-macbert'))
        model.load_state_dict(torch.load(p, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        single_pred(model, val_loader, pred_save_path=os.path.join(root_path,f'pred_data/model_{i}_pred.txt'), device='cuda' if torch.cuda.is_available() else 'cpu')

    pred_path = [os.path.join(root_path,f'pred_data/model_{i}_pred.txt') for i in range(4)]
    pred_all = {}
    true_all = []
    for i, file in enumerate(pred_path):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            y_true = [int(line.split('\t')[0]) for line in lines]
            y_pred = [int(line.split('\t')[1]) for line in lines]
            pred_all[f'model_{i}'] = y_pred
            if i == 0:
                true_all = y_true
    with open(os.path.join(root_path, "pred_data/SGDClassifier_pred.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        y_pred = [int(line.split('\t')[0]) for line in lines]
        pred_all['sgd'] = y_pred

    vote_pred = []
    for i in range(len(true_all)):
        vote = Counter([pred_all['model_0'][i], pred_all['model_1'][i], pred_all['model_2'][i], pred_all['model_3'][i], pred_all['sgd'][i]])
        vote_pred.append(vote.most_common(1)[0][0])
    print(f'acc:{accuracy_score(true_all, vote_pred)} | f1:{f1_score(true_all, vote_pred)}')    # acc:0.9432142857142857 | f1:0.9548936170212766

def predict_unlabeled(model, unlabeled_loader, pred_save_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(unlabeled_loader, desc='Predicting'):
            # 根据数据加载器的结构调整解包方式
            if len(batch) == 6:
                input_ids, attention_mask, label, model_label, source_label, stat_features = [x.to(device) for x in batch]
            elif len(batch) == 4:
                input_ids, attention_mask, label, stat_features = [x.to(device) for x in batch]
                model_label = None
                source_label = None
            else:
                raise ValueError(f"Unexpected batch format with length {len(batch)}")
            
            # 模型前向传播
            if stat_features is not None:
                label_pred, _, _ = model(input_ids, attention_mask, stat_features)
            else:
                label_pred, _, _ = model(input_ids, attention_mask)
                
            preds = torch.argmax(label_pred, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())

    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as fw:
            for p in all_preds:
                fw.write(f"{p}\n")  # 只保存预测结果

    print(f'Predictions saved to {pred_save_path}')
    return all_preds

def vote_test():
    root_path = os.path.dirname(os.path.dirname(__file__))
    bert_path = os.path.normpath(os.path.join(root_path, 'cn-macbert')).replace("\\", "/")

    model_path = [os.path.join(root_path,'model_save/best_model_fold_{}.pth'.format(i)) for i in range(1,6) if i != 3]
    tokenizer = BertTokenizer.from_pretrained(bert_path, locals_file_only=True)

    val_data = read_json(os.path.join(root_path,'data/test.json'))
    val_dataset = CustomDataset(val_data, tokenizer, model_encoder=True, source_encoder=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # for i, p in enumerate(model_path):
    #     model = EnhancedMMOE(bert_path=bert_path)
    #     model.load_state_dict(torch.load(p, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    #     model.to('cuda' if torch.cuda.is_available() else 'cpu')
    #     predict_unlabeled(model, val_loader, pred_save_path=os.path.join(root_path,f'pred_data/model_{i}_pred_test.txt'), device='cuda' if torch.cuda.is_available() else 'cpu')

    pred_path = [os.path.join(root_path,f'pred_data/model_{i}_pred_test.txt') for i in range(4)]
    pred_all = {}
    for i, file in enumerate(pred_path):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            y_pred = [int(line.split('\t')[0]) for line in lines]
            pred_all[f'model_{i}'] = y_pred
    with open(os.path.join(root_path, "pred_data/SGDClassifier_pred.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        y_pred = [int(line.split('\t')[0]) for line in lines]
        pred_all['sgd'] = y_pred

    vote_pred = []
    print(len(pred_all['model_0']), len(pred_all['model_1']), len(pred_all['model_2']), len(pred_all['model_3']), len(pred_all['sgd']))
    for i in range(len(pred_all['model_0'])):
        vote = Counter([pred_all['model_0'][i], pred_all['model_1'][i], pred_all['model_2'][i], pred_all['model_3'][i], pred_all['sgd'][i]])
        vote_pred.append(vote.most_common(1)[0][0])

    with open(os.path.join(root_path, "pred_data/vote_pred_test.txt"), 'w', encoding='utf-8') as f:
        for p in vote_pred:
            f.write(f"{p}\n")

def vote_else():
    root_path = os.path.dirname(os.path.dirname(__file__))
    bert_path = os.path.normpath(os.path.join(root_path, 'cn-macbert')).replace("\\", "/")

    model_path = [os.path.join(root_path,'model_save/best_model_fold_{}.pth'.format(i)) for i in range(1,6) if i != 3]
    tokenizer = BertTokenizer.from_pretrained(bert_path, locals_file_only=True)

    val_data = read_json(os.path.join(root_path,'data/train_dealed.json'))
    val_dataset = CustomDataset(val_data, tokenizer, model_encoder=True, source_encoder=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for i, p in enumerate(model_path):
        model = EnhancedMMOE(bert_path=bert_path)
        model.load_state_dict(torch.load(p, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        predict_unlabeled(model, val_loader, pred_save_path=os.path.join(root_path,f'pred_data/model_{i}_pred_train.txt'), device='cuda' if torch.cuda.is_available() else 'cpu')

    pred_path = [os.path.join(root_path,f'pred_data/model_{i}_pred_train.txt') for i in range(4)]
    pred_all = {}
    for i, file in enumerate(pred_path):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            y_pred = [int(line.split('\t')[0]) for line in lines]
            pred_all[f'model_{i}'] = y_pred
    with open(os.path.join(root_path, "pred_data/1.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        y_pred = [int(line.split('\t')[0]) for line in lines]
        pred_all['sgd'] = y_pred

    vote_pred = []
    print(len(pred_all['model_0']), len(pred_all['model_1']), len(pred_all['model_2']), len(pred_all['model_3']), len(pred_all['sgd']))
    for i in range(len(pred_all['model_0'])):
        vote = Counter([pred_all['model_0'][i], pred_all['model_1'][i], pred_all['model_2'][i], pred_all['model_3'][i], pred_all['sgd'][i]])
        vote_pred.append(vote.most_common(1)[0][0])

    with open(os.path.join(root_path, "pred_data/vote_pred_train.txt"), 'w', encoding='utf-8') as f:
        for p in vote_pred:
            f.write(f"{p}\n")

    true_all = []
    with open(r'G:\LLM_detection\NLPCC-2025-Task1-main\ours_model\model_0_pred.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        y_true = [int(line.split('\t')[0]) for line in lines]
        true_all = y_true
    f1 = f1_score(true_all, vote_pred)
    acc = accuracy_score(true_all, vote_pred)
    print(f'F1: {f1}, Acc: {acc}')

def save2json(txt_path, json_path, save_path):
    import json

    # 1. 读取原始 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)  # 加载已有数据

    # 2. 读取 labels.txt
    with open(txt_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]  # 去除空行和换行符

    # 3. 检查数据长度是否匹配
    if len(json_data) != len(labels):
        raise ValueError(f"标签数量 ({len(labels)}) 与 JSON 数据数量 ({len(json_data)}) 不匹配！")

    # 4. 将标签添加到 JSON 数据
    for item, label in zip(json_data, labels):
        item["label"] = label  # 添加 label 字段

    # 5. 保存为新的 JSON 文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)  # 美化输出

    print(f"标签添加完成！结果已保存到 {save_path}。")   

if __name__ == '__main__':
    # vote_dev()
    # vote_test()
    # vote_else()

    root = os.path.dirname(os.path.dirname(__file__))
    txt_path = os.path.join(root, 'pred_data/vote_pred_test.txt')
    json_path = os.path.join(root, 'data/test.json')
    save_path = os.path.join(root, 'data/test_pred.json')
    save2json(txt_path=txt_path, json_path=json_path, save_path=save_path)
    