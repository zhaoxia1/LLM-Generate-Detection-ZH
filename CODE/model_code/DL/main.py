from model import EnhancedMMOE
from train_eval import ensemble_inference
from feature_extractor import CustomDataset
import os, torch, warnings, random
from feature_extractor import read_json
from transformers import BertTokenizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from transformers import get_cosine_schedule_with_warmup
from train_eval import train, evaluate, ensemble_inference
import torch.nn.functional as F 

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置随机种子
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 路径
data_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def main(train_file, test_file, bert_path, num_epochs=10, batch_size=32, num_unfreeze_layers=4, save_folder='model_save'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    train_data = read_json(train_file)
    test_data = read_json(test_file)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # ### BEST MODEL MODIFIED: 构建测试集（用于最后集成推理）
    train_dataset_tmp = CustomDataset(train_data, tokenizer)
    test_dataset = CustomDataset(test_data, tokenizer, model_encoder=train_dataset_tmp.model_encoder,
                                 source_encoder=train_dataset_tmp.source_encoder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_losses = []
    f1_scores = []
    acc_scores = []
    val_losses = []

    ensemble_models = []  # 用于保存每折效果最好的模型

    for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
        print(f'Fold {fold + 1}')
        train_fold = [train_data[i] for i in train_index]
        val_fold = [train_data[i] for i in val_index]

        train_dataset = CustomDataset(train_fold, tokenizer)
        val_dataset = CustomDataset(val_fold, tokenizer, model_encoder=train_dataset.model_encoder,
                                    source_encoder=train_dataset.source_encoder)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = EnhancedMMOE(bert_path)
        optimizer = torch.optim.AdamW([
            {"params": model.bert.encoder.layer[-num_unfreeze_layers:].parameters(), "lr": 2e-4},
            {"params": model.bert.pooler.parameters(), "lr": 2e-4},
            # {"params": model.transformer_encoder.parameters(), "lr": 2e-4},
            {"params": model.shared_experts.parameters(), "lr": 1e-4},
            {"params": model.task_experts.parameters(), "lr": 1e-4},
            {"params": model.gate_networks.parameters(), "lr": 1e-4},
            {"params": model.label_head.parameters(), "lr": 2e-3},
            {"params": model.model_head.parameters(), "lr": 2e-3},
            {"params": model.source_head.parameters(), "lr": 2e-3},
            {"params": model.stat_feature_transform.parameters(), "lr": 1e-4}
        ])

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )
        scaler = GradScaler()

        fold_loss = []
        f1_epochs = []
        acc_epochs = []
        val_fold_loss = []

        best_val_loss = float('inf')  # ### BEST MODEL MODIFIED: 初始化最佳验证损失
        best_f1 = 0.0  # ### BEST MODEL MODIFIED: 初始化最佳F1分数
        best_model_state = None  # ### BEST MODEL MODIFIED: 初始化最佳模型

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, scheduler, scaler, epoch + 1)
            fold_loss.append(train_loss)
            f1, acc = evaluate(model, test_loader, f=fold, e=epoch)
            # f1, acc = evaluate(model, val_loader)
            f1_epochs.append(f1)
            acc_epochs.append(acc)

            # 计算验证集损失
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
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

                    label_out, model_out, source_out = model(input_ids, attention_mask, stat_features)
                    # label_out, model_out, source_out = model(input_ids, attention_mask)
                    label_loss = F.cross_entropy(label_out, label,
                                                 weight=torch.tensor([1, 1], dtype=torch.float32).to(device))
                    if model_label is not None:
                        model_loss = F.cross_entropy(model_out, model_label)
                        source_loss = F.cross_entropy(source_out, source_label)
                        # loss = 0.7 * label_loss + 0.5 * model_loss + 0.3 * source_loss
                        loss = label_loss
                    else:
                        loss = label_loss

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_fold_loss.append(avg_val_loss)
            print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(save_folder, f'best_model_fold_{fold + 1}.pth'))
                print(f'---> New best model saved at epoch {epoch + 1} with validation loss {avg_val_loss} and F1 score {f1}')

        best_model = EnhancedMMOE(bert_path)
        best_model.load_state_dict(torch.load(os.path.join(save_folder,f'best_model_fold_{fold + 1}.pth')))
        best_model.to(device)
       

        # 评估最佳模型在验证集和测试集上的性能
        val_f1, val_acc = evaluate(best_model, val_loader)
        test_f1, test_acc = evaluate(best_model, test_loader)

        f1_scores.append(test_f1)
        acc_scores.append(test_acc)
        val_losses.append(best_val_loss)

        print(f'Fold {fold + 1} - Validation F1: {val_f1}, Validation Acc: {val_acc}, Validation Loss: {best_val_loss}')
        print(f'Fold {fold + 1} - Test F1: {test_f1}, Test Acc: {test_acc}')



     ### BEST MODEL MODIFIED: 加载每折最佳模型进行集成推理
    ensemble_models = []
    for fold in range(5):
        model = EnhancedMMOE(bert_path)
        model.load_state_dict(torch.load(os.path.join(save_folder,f'best_model_fold_{fold + 1}.pth')))
        model.to(device)
        ensemble_models.append(model)
        print(f'Loaded best model from fold {fold + 1}')

    print('f1_scores:\n', f1_scores)
    print('acc_scores:\n', acc_scores)

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_acc = sum(acc_scores) / len(acc_scores)
    print(f'Average F1 Score on Test Set: {avg_f1}, Average Accuracy on Test Set: {avg_acc}')

    max_f1 = max(f1_scores)
    max_acc = max(acc_scores)
    print(f'Max F1 Score on Test Set: {max_f1}, Max Accuracy on Test Set: {max_acc}')

    ### BEST MODEL MODIFIED: 使用集成最优模型对测试集进行投票推理
    ensemble_preds, ensemble_f1, ensemble_acc = ensemble_inference(ensemble_models, test_loader)
    print(f'Ensemble Voting - F1 Score: {ensemble_f1}, Accuracy: {ensemble_acc}')
    with open(os.path.join(data_path,"pred_data/ensemble_predictions.txt"), "w", encoding="utf-8") as fout:
        fout.write("label\tpredict\n")
        for true_item, pred in zip(test_dataset.data, ensemble_preds):
            fout.write(f"{true_item['label']}\t{pred}\n")

    # 绘制训练损失曲线
    print('val_losses:\n', val_losses)
    # for i, val_fold_loss in enumerate(val_losses):
    #     plt.plot(range(num_epochs), val_fold_loss, label=f'Fold {i + 1}')
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Loss')
    # plt.title('Validation Loss per Fold')
    # plt.legend()
    # plt.savefig('validation_loss_curve.png')
    # plt.show()


if __name__ == "__main__":
    save_folder = os.path.join(data_path, 'model_save')
    train_file = os.path.join(data_path, "data/train_dealed.json")
    test_file = os.path.join(data_path, "data/dev_dealed.json")
    bert_path = os.path.join(data_path, "cn-macbert")
    main(train_file, test_file, bert_path, num_epochs=15, batch_size=32, num_unfreeze_layers=4, save_folder=save_folder)