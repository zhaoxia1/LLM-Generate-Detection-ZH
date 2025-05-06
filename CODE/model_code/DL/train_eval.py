import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from statistics import mode

# 训练函数
def train(model, train_loader, optimizer, scheduler, scaler, epoch, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.train()
    total_loss = 0
    train_loader = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    for batch in train_loader:
        if len(batch) == 6:
            input_ids, attention_mask, label, model_label, source_label, stat_features = batch
        elif len(batch) == 4:
            input_ids, attention_mask, label, stat_features = batch
            model_label = None
            source_label = None

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        stat_features = stat_features.to(device)
        if model_label is not None:
            model_label = model_label.to(device)
            source_label = source_label.to(device)

        optimizer.zero_grad()

        with autocast():
            ### BEST MODEL MODIFIED: 传入 stat_features
            label_out, model_out, source_out = model(input_ids, attention_mask, stat_features)
            label_loss = F.cross_entropy(label_out, label, weight=torch.tensor([3, 1], dtype=torch.float32).to(device))
            if model_label is not None:
                model_loss = F.cross_entropy(model_out, model_label)
                source_loss = F.cross_entropy(source_out, source_label)
                # loss = 0.5 * label_loss + 0.3 * model_loss + 0.2 * source_loss
                loss = 0.7*label_loss + 0.3*model_loss + 0.2*source_loss
            else:
                loss = label_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        train_loader.set_postfix({'Loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}, Train Loss: {avg_loss}')
    return avg_loss


# 评估函数
def evaluate(model, val_loader, f=0, e=0, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluation'):
            if len(batch) == 6:
                input_ids, attention_mask, label, model_label, source_label, stat_features = [x.to(device) for x in
                                                                                              batch]
            elif len(batch) == 4:
                input_ids, attention_mask, label, stat_features = [x.to(device) for x in batch]
                model_label = None
                source_label = None
            label_pred, _, _ = model(input_ids, attention_mask, stat_features)
            preds = torch.argmax(label_pred, dim=1).cpu().numpy()
            all_labels.extend(label.tolist())
            all_preds.extend(preds.tolist())
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print(f'Epoch {e + 1}, Fold {f + 1}, Val F1: {f1}, Val Acc: {acc}')
    return f1, acc


# 投票函数
def ensemble_inference(model_list, test_loader, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    for model in model_list:
        model.eval()
    all_labels = []
    ensemble_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensemble Inference"):
            # 假设测试集返回包含统计特征的 4 个元素
            input_ids, attention_mask, label, stat_features = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            stat_features = stat_features.to(device)
            # 保存每个模型的预测结果
            batch_preds = []
            for model in model_list:
                label_out, _, _ = model(input_ids, attention_mask, stat_features)
                # label_out, _, _ = model(input_ids, attention_mask)
                preds = torch.argmax(label_out, dim=1)  # (batch_size,)
                batch_preds.append(preds.cpu().tolist())
            # 对每个样本进行多数投票
            for preds in zip(*batch_preds):
                try:
                    vote = mode(preds)
                except:
                    vote = preds[0]
                ensemble_preds.append(vote)
            all_labels.extend(label.tolist())
    f1 = f1_score(all_labels, ensemble_preds)
    acc = accuracy_score(all_labels, ensemble_preds)
    return ensemble_preds, f1, acc
