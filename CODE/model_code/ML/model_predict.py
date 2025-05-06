import os, re, jieba
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import Counter
import jieba, jieba.posseg as pseg
from snownlp import SnowNLP


# 必须包含与训练时完全相同的特征提取类
class ChineseFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    提取中文文本的统计、句法、语义及信息论特征。
    """
    SENT_END_RE = re.compile(r"[。！？；…]+")
    TEMPLATE_PATTERNS = [
        r'首先，.*?其次，.*?最后',
        r'一方面，.*?另一方面',
        r'\d+个方面',
        r'总的来说.*?综上所述',
        r'总体来说.*?因此'
    ]

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _clean_text(text: str) -> str:
        # 清洗空白字符
        return text.replace('\u3000', ' ') \
            .replace('\r', ' ') \
            .replace('\n', ' ') \
            .replace('\t', ' ') \
            .strip()

    def transform(self, texts):
        records = []
        for text in tqdm(texts, desc="提取特征"):
            t = self._clean_text(text)
            stats = self._statistical(t)
            synt = self._syntactic(t)
            sema = self._semantic(t)
            llm = self._llm_features(t)
            records.append({**stats, **synt, **sema, **llm})
        return pd.DataFrame(records).fillna(0)

    def _statistical(self, text):
        words = jieba.lcut(text)
        wc = len(words)
        cc = len(text)
        sents = [s for s in self.SENT_END_RE.split(text) if s]
        return {
            'word_count': wc,
            'char_count': cc,
            'avg_word_len': cc / wc if wc else 0,
            'unique_word_ratio': len(set(words)) / wc if wc else 0,
            'digit_density': sum(c.isdigit() for c in text) / cc if cc else 0,
            'alpha_density': sum(c.isalpha() for c in text) / cc if cc else 0,
            'punct_density': sum(c in '，。！？；：、…“”（）—' for c in text) / cc if cc else 0,
            'sent_len_var': np.var([len(s) for s in sents]) if sents else 0
        }

    def _syntactic(self, text):
        wp = list(pseg.lcut(text))
        total = len(wp) or 1
        pos_counts = Counter(flag for _, flag in wp)
        sents = [s for s in self.SENT_END_RE.split(text) if s]
        long_ratio = sum(len(s) > 50 for s in sents) / len(sents) if sents else 0
        # 计算相邻句子的余弦相似度
        sims = []
        for a, b in zip(sents, sents[1:]):
            c1, c2 = Counter(jieba.lcut(a)), Counter(jieba.lcut(b))
            sims.append(self._cosine(c1, c2))

        return {
            'noun_ratio': (pos_counts.get('n', 0) + pos_counts.get('nr', 0)) / total,
            'verb_ratio': pos_counts.get('v', 0) / total,
            'adj_ratio': pos_counts.get('a', 0) / total,
            'adv_ratio': pos_counts.get('d', 0) / total,
            'pron_ratio': pos_counts.get('r', 0) / total,
            'struct_ratio': pos_counts.get('c', 0) / total,
            'passive_bei': sum(1 for word, flag in wp if word == '被') / total,
            'long_sent_ratio': long_ratio,
            'coherence_mean': np.mean(sims) if sims else 0,
            'coherence_var': np.var(sims) if sims else 0
        }

    @staticmethod
    def _cosine(c1: Counter, c2: Counter) -> float:
        inter = set(c1) & set(c2)
        num = sum(c1[x] * c2[x] for x in inter)
        den = np.sqrt(sum(v * v for v in c1.values()) * sum(v * v for v in c2.values()))
        return num / den if den else 0

    def _semantic(self, text):
        try:
            snow = SnowNLP(text)
            sentiment = snow.sentiments
            keywords = snow.keywords(limit=5)
        except Exception:
            sentiment, keywords = 0.5, []

        tokens = jieba.lcut(text)
        subj_triggers = {'认为', '觉得', '相信', '推测', '可能', '感觉', '估计', '似乎', '恐怕', '大概', '显然', '无疑',
                         '固然'}
        subjectivity = sum(tok in subj_triggers for tok in tokens) / len(tokens) if tokens else 0
        return {
            'sentiment': sentiment,
            'subjectivity': subjectivity,
            'keywords_count': len(keywords)
        }

    def _llm_features(self, text):
        tokens = jieba.lcut(text)
        total = len(tokens)
        freq = Counter(tokens)
        # 信息论特征
        if total:
            entropy = -sum((cnt / total) * np.log(cnt / total) for cnt in freq.values())
            perplexity = np.exp(entropy)
        else:
            entropy, perplexity = 0, 1

        # 重复率
        k = 8
        segments = [text[i:i + k] for i in range(len(text) - k + 1)] or ['']
        repetition = max(Counter(segments).values()) / len(segments)
        # 模板句式计数
        template_count = sum(len(re.findall(pat, text)) for pat in self.TEMPLATE_PATTERNS)
        bigrams = [tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0

        return {
            'entropy': entropy,
            'perplexity': perplexity,
            'repetition': repetition,
            'template_count': template_count,
            'bigram_diversity': bigram_diversity
        }

def load_artifacts(model_dir="model_save"):
    """加载所有必需的预处理对象和模型"""
    artifacts = {
        'extractor': ChineseFeatureExtractor(),
        'selector': joblib.load(os.path.join(model_dir, "selector_custom.pkl")),
        'models': {
            'SGDClassifier': joblib.load(os.path.join(model_dir, "SGDClassifier.joblib")),
            'NaiveBayes': joblib.load(os.path.join(model_dir, "NaiveBayes.joblib")),
            'LightGBM': joblib.load(os.path.join(model_dir, "lgbm_model.joblib")),
            'CatBoost': joblib.load(os.path.join(model_dir, "catboost_model.joblib")),
            'XGBoost': joblib.load(os.path.join(model_dir, "xgb_model.joblib"))
        }
    }
    
    return artifacts

def predict(texts: Union[str, List[str]], artifacts: dict, save_path = None) -> np.ndarray:
    """
    预测函数
    :param texts: 要预测的文本（支持单个字符串或列表）
    :param artifacts: 通过load_artifacts加载的预处理对象和模型
    :return: 预测结果数组
    """
    # 统一输入格式
    if isinstance(texts, str):
        texts = [texts]
    
    # 1. 特征提取
    print('开始提取特征...')
    try:
        df_features = np.load(save_path)
        print('加载特征成功!')
    except:
        df_features = artifacts['extractor'].transform(texts)
        if save_path is not None:
            np.save(save_path, df_features)

    
    # 2. 特征选择
    selected_features = artifacts['selector'].transform(df_features)
    
    # 3. 模型投票预测
    return voting(artifacts['models'], selected_features)

def voting(models: dict, X: np.ndarray) -> np.ndarray:
    """投票逻辑（保持与训练代码相同）"""
    predictions = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            predictions.append(np.argmax(proba, axis=1))
        else:
            predictions.append(model.predict(X))
    
    # 硬投票
    return np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=np.array(predictions)
    )

def evaluate_individual_models(texts: Union[str, List[str]], y_true: list, artifacts: dict) -> dict:
    """
    评估每个单独模型的性能指标
    :param texts: 要评估的文本（需与y_true顺序一致）
    :param y_true: 真实标签列表
    :param artifacts: 通过load_artifacts加载的预处理对象和模型
    :return: 包含各模型评估指标的字典
    """
    # 统一输入格式
    if isinstance(texts, str):
        texts = [texts]
    
    # 生成特征数据（与预测流程保持一致）
    df_features = artifacts['extractor'].transform(texts)

    # folder_path = os.path.dirname(os.path.dirname(__file__))
    # X_dev_feat = np.load(os.path.join(folder_path, 'Features/X_dev_feat.npy'))
    # print('开始提取验证集特征')
    # X_dev_feat = extractor.transform(X_dev)
    # y_true = np.load(os.path.join(folder_path, 'Features/y_dev.npy'))
    
    selected_features = artifacts['selector'].transform(df_features)

    # 初始化评估结果存储
    model_scores = {}

    # 遍历每个模型进行评估
    for model_name, model in artifacts['models'].items():
        try:
            # 获取预测结果
            y_pred = model.predict(selected_features)
            
            # 计算各项指标
            model_scores[model_name] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
        except Exception as e:
            print(f"评估 {model_name} 时出现异常: {str(e)}")
            model_scores[model_name] = None

    return model_scores

def predict_single_models(texts: Union[str, List[str]], 
                         artifacts: dict, 
                         save_dir: str = "single_model_preds",
                         features_path = None) -> dict:
    """
    单模型独立预测并保存结果
    
    :param texts: 要预测的文本（支持单个字符串或列表）
    :param artifacts: 通过load_artifacts加载的预处理对象和模型
    :param save_dir: 预测结果保存目录
    :return: 包含各模型预测结果的字典
    """
    # 统一输入格式
    if isinstance(texts, str):
        texts = [texts]

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 特征提取
    print('开始提取特征...')
    try:
        df_features = np.load(features_path)
        print('加载特征成功!')
    except:
        df_features = artifacts['extractor'].transform(texts)
        if features_path is not None:
            np.save(features_path, df_features)
    
    # 2. 特征选择
    selected_features = artifacts['selector'].transform(df_features)
    
    # 3. 各模型独立预测
    all_preds = {}
    
    for model_name, model in artifacts['models'].items():
        try:
            # 执行预测
            if hasattr(model, 'predict_proba'):
                y_pred = np.argmax(model.predict_proba(selected_features), axis=1)
            else:
                y_pred = model.predict(selected_features)
            
            # 存储结果
            all_preds[model_name] = y_pred
            
            # 保存到文件
            save_path = os.path.join(save_dir, f"{model_name}_pred.txt")
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(map(str, y_pred)))
            
            print(f"{model_name} 预测结果已保存至 {save_path}")
            
        except Exception as e:
            print(f"模型 {model_name} 预测失败: {str(e)}")
            all_preds[model_name] = None
    
    return all_preds

def predict_test():
    folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    feature_path = os.path.join(folder_path, 'Features/X_test_feat.npy')

    # 加载预处理对象和模型
    artifacts = load_artifacts(model_dir=os.path.join(folder_path,"model_save"))
    
    json_data = pd.read_json(os.path.join(folder_path,"data/test.json"), lines=False)
    texts = json_data['text']
    
      # 集成模型预测
    print("执行集成模型投票预测...")
    voting_pred = predict(texts, artifacts, save_path=feature_path)
    with open(os.path.join(folder_path,'pred_data/vote_pred.txt'),'w',encoding='utf-8') as f:
        f.write("\n".join(map(str, voting_pred)))
    
    # 单模型独立预测
    print("\n执行单模型独立预测...")
    save_dir = os.path.join(folder_path,'pred_data')
    single_preds = predict_single_models(texts, artifacts, save_dir=save_dir, features_path=feature_path)
    
    return voting_pred, single_preds
            

# 使用示例
def predict_dev():
    folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    feature_path = os.path.join(folder_path, 'Features/X_dev_feat_all.npy')

    # 加载预处理对象和模型
    artifacts = load_artifacts(model_dir=os.path.join(folder_path,"model_save"))
    
    json_data = pd.read_json(os.path.join(folder_path, 'data/dev_dealed.json'), lines=False)
    text = json_data['text']
    all_label = np.load(os.path.join(folder_path,'Features/y_dev_all.npy'))
    
    # 执行预测
    pred_label = predict(text, artifacts, save_path=feature_path)

    print("准确率：", accuracy_score(all_label, pred_label))
    print("精确率：", precision_score(all_label, pred_label, average='weighted'))
    print("召回率：", recall_score(all_label, pred_label, average='weighted'))
    print("F1值：", f1_score(all_label, pred_label, average='weighted'))

    # 评估单个模型
    individual_scores = evaluate_individual_models(text, all_label, artifacts)
    
    print("\n=== 各模型独立评估结果 ===")
    for model_name, scores in individual_scores.items():
        print(f"\n{model_name:-^40}")
        print(f"准确率：{scores['accuracy']:.4f}")
        print(f"精确率：{scores['precision']:.4f}") 
        print(f"召回率：{scores['recall']:.4f}")
        print(f"F1值：{scores['f1']:.4f}")


if __name__ == '__main__':
    # predict_dev()
    predict_test()