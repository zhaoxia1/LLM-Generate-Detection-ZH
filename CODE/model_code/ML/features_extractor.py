import re, os, warnings, joblib
from collections import Counter
import jieba, jieba.posseg as pseg
import numpy as np
import pandas as pd
from tqdm import tqdm
from snownlp import SnowNLP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 集成投票
def voting(models, X):
    predictions = []
    # 遍历每个模型进行预测
    for model in models.values():
        y_pred = model.predict(X)
        predictions.append(y_pred)

    predictions = np.array(predictions)
    final_predictions = []
    # 对每个样本进行投票
    for i in range(predictions.shape[1]):
        sample_predictions = predictions[:, i]
        # 统计每个类别的票数
        unique_classes, counts = np.unique(sample_predictions, return_counts=True)
        # 选取票数最多的类别
        final_class = unique_classes[np.argmax(counts)]
        final_predictions.append(final_class)

    return np.array(final_predictions)

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


def main():
    folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print('folder_path: ', folder_path)
    train_df = pd.read_json(os.path.join(folder_path, 'data/train_dealed.json'), orient='records')
    dev_df = pd.read_json(os.path.join(folder_path, 'data/dev_dealed.json'), orient='records')

    train_sample_size = 6000  # 采样大小
    dev_sample_size = 1000  # 采样大小
    feature_num = 15  # 选择特征数量
    load_saved_features = False  # 是否加载保存的特征, 运行一次后可将此处改为True
    use_svd = False  # 是否降维
    use_tfidf = False  # 是否使用TF-IDF特征


    X_train, y_train = train_df['text'].tolist()[:train_sample_size], train_df['label'].tolist()[:train_sample_size]
    X_dev, y_dev = dev_df['text'].tolist()[:dev_sample_size], dev_df['label'].tolist()[:dev_sample_size]

    # 提取自定义特征
    if not load_saved_features:
        extractor = ChineseFeatureExtractor()
        print('开始提取训练集特征')
        X_train_feat = extractor.fit_transform(X_train)
        print('开始提取验证集特征')
        X_dev_feat = extractor.transform(X_dev)

        # 保存提取的特征
        np.save(os.path.join(folder_path, 'Features/X_train_feat_all.npy'), X_train_feat)
        np.save(os.path.join(folder_path, 'Features/y_train_all.npy'), y_train)
        np.save(os.path.join(folder_path, 'Features/X_dev_feat_all.npy'), X_dev_feat)
        np.save(os.path.join(folder_path, 'Features/y_dev_all.npy'), y_dev)

    else:
        # 加载特征,使用此部分代码需注释后面 特征数据 保存部分代码
        print('加载保存的特征...')
        X_train_feat = np.load(os.path.join(folder_path, 'Features/X_train_feat_all.npy'))
        y_train = np.load(os.path.join(folder_path, 'Features/y_train_all.npy'))
        X_dev_feat = np.load(os.path.join(folder_path, 'Features/X_dev_feat_all.npy'))
        y_dev = np.load(os.path.join(folder_path, 'Features/y_dev_all.npy'))

    # 使用过滤法选择自定义特征中的前k个
    selector_custom = SelectKBest(score_func=f_classif, k=feature_num)
    X_train_custom_selected = selector_custom.fit_transform(X_train_feat, y_train)
    # print(f'训练集选择的特征: \n{X_train_custom_selected}')
    X_dev_custom_selected = selector_custom.transform(X_dev_feat)
    # print(f'验证集选择的特征: \n{X_dev_custom_selected}')


    if use_tfidf:
        # 创建TF-IDF向量化器
        tfidf_vectorizer = TfidfVectorizer()
        # 对训练集文本进行TF-IDF特征提取
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        print(f'Train TF-IDF shape: {X_train_tfidf.shape}')
        # 对验证集文本进行TF-IDF特征提取
        X_dev_tfidf = tfidf_vectorizer.transform(X_dev)
        print(f'Dev TF-IDF shape: {X_dev_tfidf.shape}')

        # 合并选择后的自定义特征和TF-IDF特征
        X_train_combined = np.hstack((X_train_custom_selected, X_train_tfidf.toarray()))
        X_dev_combined = np.hstack((X_dev_custom_selected, X_dev_tfidf.toarray()))

    # 降维
    if use_svd:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=1000)  # 设置降维后的维度为1000
        X_train_tfidf = svd.fit_transform(X_train_tfidf)
        X_dev_tfidf = svd.transform(X_dev_tfidf)
        print(f'Train TF-IDF with SVD shape: {X_train_tfidf.shape}')
        print(f'Dev TF-IDF with SVD shape: {X_dev_tfidf.shape}')

    # 赋值
    if use_tfidf:
        X_train_selected = X_train_combined
        X_dev_selected = X_dev_combined
    else:
        X_train_selected = X_train_custom_selected
        X_dev_selected = X_dev_custom_selected

    # 定义各基模型
    sgd_clf = Pipeline([
        ('scale', StandardScaler()),
        ('clf', SGDClassifier(
            loss='log_loss', max_iter=1000, tol=1e-3,
            learning_rate='optimal', n_jobs=-1, random_state=42
        ))
    ])
    nb_clf = MultinomialNB()
    lgbm_clf = LGBMClassifier(
        n_estimators=1000, learning_rate=0.05,
        max_depth=6, num_leaves=31, max_bin=63,
        subsample=0.8, feature_fraction=0.8, bagging_freq=1,
        n_jobs=-1, random_state=42, verbose=-1
    )
    cb_clf = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        subsample=0.8, rsm=0.8,
        thread_count=-1, random_state=42, verbose=False
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=0,
        eval_metric='logloss'
    )

    # 各自传自己的 early stopping
    print('开始训练模型...')
    print('sgd_clf is training...')
    sgd_clf.fit(X_train_selected, y_train)
    print('nb_clf is training...')
    nb_clf.fit(X_train_selected, y_train)
    print('lgbm_clf is training...')
    lgbm_clf.fit(
        X_train_selected, y_train,
        eval_set=[(X_dev_selected, y_dev)],
        eval_metric=
        'multi_logloss'
        ,
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)]
    )
    print('cb_clf is training...')
    cb_clf.fit(
        X_train_selected, y_train,
        eval_set=Pool(X_dev_selected, y_dev),
        use_best_model=True,
        early_stopping_rounds=50,
        verbose=False
    )
    print('xgb_clf is training...')
    xgb_clf.fit(
        X_train_selected, y_train,
        eval_set=[(X_dev_selected, y_dev)],
        verbose = False
    )

    print("\n=== Soft Voting 投票过程中各模型评估 ===")

    # 构建模型字典
    models = {
        'SGDClassifier': sgd_clf,
        'NaiveBayes': nb_clf,
        'LightGBM': lgbm_clf,
        'CatBoost': cb_clf,
        'XGBoost': xgb_clf
    }

    vote_results = {}

    # 保存模型
    model_dir = os.path.join(folder_path,"model_save")
    for name, model in models.items():
        # 通用保存方法（适用于所有scikit-learn兼容模型）
        joblib.dump(model, f"{model_dir}/{name}.joblib")
        
        prob = model.predict_proba(X_dev_selected)
        pred = np.argmax(prob, axis=1)
        acc = accuracy_score(y_dev, pred)
        f1 = f1_score(y_dev, pred, average='weighted')
        vote_results[name] = {'accuracy': acc, 'f1': f1}
        print(f"{name:<12} | Accuracy = {acc:.4f}, F1 = {f1:.4f}")

    # 在投票中效果最好的模型
    best_model_name = max(vote_results, key=lambda k: vote_results[k]['f1'])
    best = vote_results[best_model_name]

    print(f"\n按模型性能投票中,效果最好的模型是: {best_model_name}")
    print(f"Accuracy = {best['accuracy']:.4f}")
    print(f"F1 Score = {best['f1']:.4f}")

     # 使用投票函数获得最终预测结果
    final_predictions = voting(models, X_dev_selected)
    vote_acc = accuracy_score(y_dev, final_predictions)
    vote_f1 = f1_score(y_dev, final_predictions, average='weighted')

    print("\n=== 投票结果评估 ===")
    print(f"投票后的 Accuracy = {vote_acc:.4f}")
    print(f"投票后的 F1 = {vote_f1:.4f}")

if __name__ == '__main__':
    main()
