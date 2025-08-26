import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import os
import torch.nn.functional as F
import random
from datetime import datetime
from glob import glob  # 用于扫描文件

# ------------------------------
# 模型与损失函数定义
# ------------------------------
class FocalLoss(nn.Module):
    """多类别Focal Loss实现"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = F.nll_loss(
            log_pt, 
            target, 
            weight=self.weight, 
            reduction=self.reduction, 
            ignore_index=self.ignore_index
        )
        return loss

class BertForSequenceClassification(nn.Module):
    """BERT分类模型（使用最后5层CLS拼接）- 用于RoBERTa系列"""
    def __init__(self, bert_model_name, num_labels=10, dropout=0.1):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size * 5, num_labels)  # 5层CLS拼接

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        hidden_states = outputs[2]  # 所有隐藏层输出
        
        # 取最后5层的CLS token
        cls_list = [hidden_states[i][:, 0, :] for i in range(-1, -6, -1)]
        last_hidden = torch.cat(cls_list, dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits


# ------------------------------
# 数据处理工具
# ------------------------------
def truncate_with_head_tail(text, tokenizer, max_length=512, head_length=128, tail_length=382):
    """文本截断策略：保留头部和尾部"""
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False
    )
    input_ids = encoded['input_ids']
    usable_length = max_length - 2  # 预留CLS和SEP
    
    if len(input_ids) <= usable_length:
        final_input_ids = input_ids
    else:
        head_ids = input_ids[:head_length]
        tail_ids = input_ids[-tail_length:] if tail_length > 0 else []
        final_input_ids = head_ids + tail_ids
    
    final_input_ids = [tokenizer.cls_token_id] + final_input_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(final_input_ids)
    padding_length = max_length - len(final_input_ids)
    
    if padding_length > 0:
        final_input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
    
    token_type_ids = [0] * max_length
    return {
        'input_ids': torch.tensor(final_input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'token_type_ids': torch.tensor(token_type_ids)
    }

class TestDataset(Dataset):
    """测试数据集（无标签）"""
    def __init__(self, texts, tokenizer, max_length=512, head_len=128, tail_len=382):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_len = head_len
        self.tail_len = tail_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = truncate_with_head_tail(
            text, self.tokenizer,
            max_length=self.max_length,
            head_length=self.head_len,
            tail_length=self.tail_len
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids']
        }

# ------------------------------
# 集成核心函数
# ------------------------------
def get_label(data):
    """获取标签映射关系"""
    unique_labels = sorted(data['类别'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label

def predict_ensemble(test_dataloader, models, model_names, device):
    """获取每个模型的详细预测结果（ID和概率）"""
    model_results = [{'name': name, 'preds': [], 'probs': []} for name in model_names]
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="集成预测中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            for i, model in enumerate(models):
                logits = model(input_ids, attention_mask, token_type_ids)
                probs = torch.softmax(logits, dim=1)
                pred_id = torch.argmax(probs, dim=1).item()
                pred_prob = probs[0, pred_id].item()
                
                model_results[i]['preds'].append(pred_id)
                model_results[i]['probs'].append(pred_prob)
    
    return model_results

def vote_preds_with_probs(model_results, id_to_label):
    """基础投票+概率加权预测"""
    num_samples = len(model_results[0]['preds'])
    final_preds = []
    
    for sample_idx in range(num_samples):
        preds = [model_res['preds'][sample_idx] for model_res in model_results]
        probs = [model_res['probs'][sample_idx] for model_res in model_results]
        
        label_stats = defaultdict(lambda: {'count': 0, 'probs': []})
        for pred_id, prob in zip(preds, probs):
            label_stats[pred_id]['count'] += 1
            label_stats[pred_id]['probs'].append(prob)
        
        max_count = max(stats['count'] for stats in label_stats.values())
        candidates = [label_id for label_id, stats in label_stats.items() if stats['count'] == max_count]
        
        if len(candidates) == 1:
            final_id = candidates[0]
        else:
            avg_probs = {label_id: np.mean(label_stats[label_id]['probs']) for label_id in candidates}
            final_id = max(avg_probs, key=avg_probs.get)
        
        final_preds.append(id_to_label[final_id])
    
    return final_preds

def ensemble_selected_models(data, selected_models, strategy='weighted_prob', top_n=4):
    """单策略集成函数（被多策略集成调用）"""
    valid_models = []
    for model in selected_models:
        pred_col = f"{model}_pred"
        prob_col = f"{model}_prob"
        if pred_col in data.columns and prob_col in data.columns:
            valid_models.append(model)
        else:
            print(f"警告: 模型 {model} 缺少预测列，已跳过")
    
    if not valid_models:
        raise ValueError("没有有效的模型用于集成")
    
    model_weights = None
    if strategy == 'model_weighted':
        model_weights = {}
        for model in valid_models:
            f1_value = float(model.split('_')[1]) if 'valF1' in model else 1.0  # 兼容不同模型名格式
            model_weights[model] = f1_value
        total_weight = sum(model_weights.values())
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    final_preds = []
    for idx, row in data.iterrows():
        model_preds = []
        model_probs = []
        model_names = []
        
        for model in valid_models:
            try:
                pred = row[f"{model}_pred"]
                prob = float(row[f"{model}_prob"])
                model_preds.append(pred)
                model_probs.append(prob)
                model_names.append(model)
            except Exception as e:
                print(f"警告: 处理样本 {idx} 的模型 {model} 时出错: {str(e)}")
        
        if not model_preds:
            final_preds.append(None)
            continue
        
        if strategy == 'voting':
            label_counts = defaultdict(int)
            for pred in model_preds:
                label_counts[pred] += 1
            final_pred = max(label_counts, key=label_counts.get)
        
        elif strategy == 'weighted_prob':
            prob_scores = defaultdict(float)
            for pred, prob in zip(model_preds, model_probs):
                prob_scores[pred] += prob
            final_pred = max(prob_scores, key=prob_scores.get)
        
        elif strategy == 'top_n_voting':
            top_n = min(top_n, len(model_probs))
            sorted_pairs = sorted(zip(model_probs, model_preds), reverse=True, key=lambda x: x[0])
            top_preds = [p for (_, p) in sorted_pairs[:top_n]]
            label_counts = defaultdict(int)
            for pred in top_preds:
                label_counts[pred] += 1
            final_pred = max(label_counts, key=label_counts.get)
        
        elif strategy == 'model_weighted':
            prob_scores = defaultdict(float)
            for i, pred in enumerate(model_preds):
                prob_scores[pred] += model_probs[i] * model_weights[model_names[i]]
            final_pred = max(prob_scores, key=prob_scores.get)
        
        elif strategy == 'prob_rank':
            sorted_pairs = sorted(zip(model_preds, model_probs), key=lambda x: x[1], reverse=True)
            rank_scores = defaultdict(float)
            for i, (pred, _) in enumerate(sorted_pairs):
                rank_scores[pred] += 1 - (i / len(sorted_pairs))
            final_pred = max(rank_scores, key=rank_scores.get)
        
        else:
            raise ValueError(f"不支持的策略: {strategy}")

        final_preds.append(final_pred)
    
    return final_preds, valid_models

def ensemble_all_strategies(data, selected_models, top_n=4):
    """多策略集成函数，生成所有策略结果"""
    strategies = [
        'voting', 
        'weighted_prob', 
        'top_n_voting', 
        'model_weighted', 
        'prob_rank'
    ]
    
    all_predictions = {}
    all_fake_labels = defaultdict(dict)
    
    for strategy in strategies:
        print(f"\n正在执行 {strategy} 策略集成...")
        preds, valid_models = ensemble_selected_models(
            data=data,
            selected_models=selected_models,
            strategy=strategy,
            top_n=top_n
        )
        all_predictions[strategy] = preds
        
        for idx, row in data.iterrows():
            model_preds = []
            model_probs = []
            for model in valid_models:
                try:
                    model_preds.append(row[f"{model}_pred"])
                    model_probs.append(float(row[f"{model}_prob"]))
                except:
                    continue
            if not model_preds:
                continue
                
    result_df = data.copy()
    
    # 添加所有策略预测结果
    for strategy, preds in all_predictions.items():
        result_df[f'ensemble_{strategy}'] = preds
    
    # 计算策略间投票的最终结果
    final_preds = []
    for idx, row in result_df.iterrows():
        strategy_preds = [row[f'ensemble_{s}'] for s in strategies if pd.notna(row[f'ensemble_{s}'])]
        if not strategy_preds:
            final_preds.append(None)
            continue
        pred_counts = defaultdict(int)
        for pred in strategy_preds:
            pred_counts[pred] += 1
        final_pred = max(pred_counts, key=pred_counts.get)
        final_preds.append(final_pred)
    
    result_df['ensemble_final'] = final_preds
    return result_df

# ------------------------------
# 主预测流程
# ------------------------------
if __name__ == "__main__":
    # 1. 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model_configs = [
        {
            "base_model": "/root/lanyun-fs/models/chinese-roberta-wwm-ext",
            "model_dirs": ["./model_path/focal_loss", "./model_path/rdrop", "./model_path/focal_loss_rdrop", "./model_path/rdrop_with_label_smooth"]
        },
    ]

    # 数据配置
    TEST_FILE = 'dataset/test_text.csv'
    BASE_OUTPUT_FILE = 'dataset/test_predictions_detailed_all_data.csv'  # 基础模型预测结果
    MAX_LENGTH = 512
    DROPOUT = 0.2
    BATCH_SIZE = 1
    TOP_N = 5
    
    # 2. 加载标签映射
    train_data = pd.read_csv('dataset/train_all.csv')
    label_to_id, id_to_label = get_label(train_data)
    NUM_LABELS = len(label_to_id)
    print(f"类别映射: {id_to_label}")
    
    # 3. 加载测试数据
    test_data = pd.read_csv(TEST_FILE)
    test_texts = test_data['文本'].values
    print(f"测试样本数量: {len(test_texts)}")
    
    # 4. 加载模型
    models = []
    model_names = []
    
    for config in model_configs:
        base_model = config['base_model']
        base_model_name = os.path.basename(base_model)

        if "roberta" in base_model_name.lower():
            model_class = BertForSequenceClassification
            print(f"检测到RoBERTa基础模型，使用BertForSequenceClassification")
        else:
            model_class = BertForSequenceClassification  # 其他模型默认使用结构1
            print(f"未识别模型 {base_model_name}，默认使用BertForSequenceClassification")
        
        for model_dir in config['model_dirs']:
            if not os.path.exists(model_dir):
                print(f"模型目录不存在: {model_dir}，已跳过")
                continue

            pth_files = glob(os.path.join(model_dir, "**", "*.pth"), recursive=True)
            if not pth_files:
                print(f"目录 {model_dir} 中未找到.pth文件，已跳过")
                continue

            for pth_path in pth_files:
                file_name = os.path.splitext(os.path.basename(pth_path))[0]
                dir_name = os.path.basename(model_dir)
                model_name = f"{base_model_name}_{dir_name}_{file_name}"

                try:
                    model = model_class(
                        bert_model_name=base_model,
                        num_labels=NUM_LABELS,
                        dropout=DROPOUT
                    ).to(device)
                    model.load_state_dict(torch.load(pth_path, map_location=device))
                    model.eval()
                    models.append(model)
                    model_names.append(model_name)
                    print(f"已加载模型: {model_name}")
                except Exception as e:
                    print(f"加载模型 {pth_path} 失败: {str(e)}，已跳过")

    print(f"\n成功加载 {len(models)} 个有效模型用于集成预测")
    if len(models) == 0:
        raise ValueError("未加载到任何模型，请检查模型目录和文件格式")
    
    # 5. 基础集成预测（单模型结果保存）
    tokenizer = BertTokenizer.from_pretrained(model_configs[0]['base_model'])
    test_dataset = TestDataset(texts=test_texts, tokenizer=tokenizer, max_length=MAX_LENGTH)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model_results = predict_ensemble(test_dataloader, models, model_names, device)
    
    final_preds = vote_preds_with_probs(model_results, id_to_label)
    result_df = test_data.copy()
    
    for model_res in model_results:
        model_name = model_res['name']
        pred_labels = [id_to_label[pred_id] for pred_id in model_res['preds']]
        result_df[f'{model_name}_pred'] = pred_labels
        result_df[f'{model_name}_prob'] = model_res['probs']
    

    result_df['base_vote_pred'] = final_preds
    result_df.to_csv(BASE_OUTPUT_FILE, index=False)
    print(f"\n基础模型预测结果已保存至: {BASE_OUTPUT_FILE}")
    print("\n所有预测与集成流程完成!")