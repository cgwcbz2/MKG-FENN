import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import os
import random

from sklearn.metrics import accuracy_score, auc, roc_auc_score, recall_score, f1_score, precision_recall_curve, \
    precision_score
from sklearn.preprocessing import label_binarize
import warnings

# 从 modeltask1.py 导入模型
from modeltask1 import FusionLayer, GNN1, GNN2, GNN3, GNN4

warnings.filterwarnings("ignore")

# --- 参数解析 (保持不变) ---
parser = argparse.ArgumentParser(description='GNN based on pre-split data for Task 1')
parser.add_argument("--epoches", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--weigh_decay", type=float, default=1e-8)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--neighbor_sample_size", type=int, default=6)
parser.add_argument("--event_num", type=int, default=65)
parser.add_argument("--n_drug", type=int, default=572)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--embedding_num", type=int, default=128)
args = parser.parse_args()


def setup_seed():
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# --- 辅助函数 (基本保持不变) ---

def read_dataset(drug_name_id, num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    filename = f"../dataset/dataset{num}.txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string = line.rstrip().split('//', 2)
            head, tail, relation = string[0], string[1], string[2]
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)

            # 使用 .get() 避免因KG文件中名称不在dict1中而报错
            head_id = drug_name_id.get(head)
            tail_id = drug_name_id.get(tail)

            if head_id is not None:
                if num == 3 and tail_id is not None:
                    kg[head_id].append((tail_id, relations[relation]))
                    kg[tail_id].append((head_id, relations[relation]))
                elif num != 3:
                    kg[head_id].append((tails[tail], relations[relation]))

    return kg, len(tails), len(relations)


# evaluate 函数保持 task1 的完整版本
def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro', zero_division=0)
    result_all[8] = precision_score(y_test, pred_type, average='macro', zero_division=0)
    result_all[9] = recall_score(y_test, pred_type, average='micro', zero_division=0)
    result_all[10] = recall_score(y_test, pred_type, average='macro', zero_division=0)
    return result_all, None  # 移除了 per-event 的评估


def roc_aupr_score(y_true, y_score, average="macro"):
    # (此函数与task2版本相同，此处省略以保持简洁)
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):
        if average == "binary": return binary_metric(y_true, y_score)
        if average == "micro":
            y_true, y_score = y_true.ravel(), y_score.ravel()
        if y_true.ndim == 1: y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1: y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def save_result(filepath, result_type, result):
    with open(filepath + result_type + 'task1.csv', "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


# --- 修改：用新的 train_and_validate 函数替换旧的 train 函数 ---
def train_and_validate(train_x, train_y, val_x, val_y, net):
    loss_function = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weigh_decay)

    train_x1 = train_x.copy()
    train_x[:, [0, 1]] = train_x[:, [1, 0]]
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y_total = torch.LongTensor(np.concatenate([train_y, train_y]))
    train_data = TensorDataset(train_x_total, train_y_total)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)

    best_val_score = 0.0
    best_model_state = None

    for epoch in range(args.epoches):
        net.train()
        train_loss_sum = 0
        train_acc_list = []
        for x, y in train_iter:
            opti.zero_grad()
            # Task1 的模型调用更简单
            output = net(x.long())
            l = loss_function(output, y.long())
            l.backward()
            opti.step()
            train_loss_sum += l.item()
            train_acc_list.append(accuracy_score(torch.argmax(output, dim=1), y))

        avg_train_loss = train_loss_sum / len(train_iter)
        avg_train_acc = np.mean(train_acc_list)

        net.eval()
        with torch.no_grad():
            val_x_tensor = torch.LongTensor(val_x)
            val_output = F.softmax(net(val_x_tensor), dim=1)
            val_label_tensor = torch.LongTensor(val_y)

            val_loss = loss_function(val_output, val_label_tensor).item()
            val_pred_np = torch.argmax(val_output, dim=1).numpy()
            val_label_np = val_label_tensor.numpy()

            val_acc = accuracy_score(val_label_np, val_pred_np)
            val_f1 = f1_score(val_label_np, val_pred_np, average='macro', zero_division=0)
            val_rec = recall_score(val_label_np, val_pred_np, average='macro', zero_division=0)
            val_pre = precision_score(val_label_np, val_pred_np, average='macro', zero_division=0)

            if val_f1 > best_val_score:
                best_val_score = val_f1
                best_model_state = net.state_dict().copy()
                print(f"Epoch {epoch + 1}: New best validation F1-score: {best_val_score:.4f}")

        print(f'Epoch [{epoch + 1}/{args.epoches}] | Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Recall: {val_rec:.4f}, Precision: {val_pre:.4f}')

    return best_model_state


# --- 修改: main 函数 ---
def main():
    # 定义CSV文件路径
    TRAIN_FILE = '../dataset/ddi_training1xiao.csv'
    VALIDATION_FILE = '../dataset/ddi_validation1xiao.csv'
    TEST_FILE = '../dataset/ddi_test1xiao.csv'

    # 1. 基础信息加载 (数据库, 知识图谱)
    conn = sqlite3.connect("../dataset/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)

    # 创建 drug name -> id 的映射字典
    dict1 = {name: i for i, name in enumerate(df_drug["name"])}
    drug_name_ids_list = list(dict1.values())

    # 创建从 DrugBank ID 到 药物全名的翻译字典
    try:
        db_id_to_name_map = pd.Series(df_drug.name.values, index=df_drug.id).to_dict()
    except AttributeError:
        print("错误: 无法在'drug'表中找到名为'id'的列。请确认列名。")
        exit()

    dataset1_kg, d1_t, d1_r = read_dataset(dict1, 1)
    dataset2_kg, d2_t, d2_r = read_dataset(dict1, 2)
    dataset3_kg, d3_t, d3_r = read_dataset(dict1, 3)
    dataset4_kg, d4_t, d4_r = read_dataset(dict1, 4)

    dataset = {"dataset1": dataset1_kg, "dataset2": dataset2_kg, "dataset3": dataset3_kg, "dataset4": dataset4_kg}
    tail_len = {"dataset1": d1_t, "dataset2": d2_t, "dataset3": d3_t, "dataset4": d4_t}
    relation_len = {"dataset1": d1_r, "dataset2": d2_r, "dataset3": d3_r, "dataset4": d4_r}

    # 2. 从CSV文件加载数据
    print("Loading data from pre-split CSV files...")

    def load_from_csv(filepath, name_to_id_map, db_id_to_name_map):
        try:
            df = pd.read_csv(filepath)
            df['drugA_name'] = df['d1'].map(db_id_to_name_map)
            df['drugB_name'] = df['d2'].map(db_id_to_name_map)
            df['drugA_id'] = df['drugA_name'].map(name_to_id_map)
            df['drugB_id'] = df['drugB_name'].map(name_to_id_map)
            df.dropna(subset=['drugA_id', 'drugB_id'], inplace=True)
            x = df[['drugA_id', 'drugB_id']].to_numpy(dtype=np.int64)
            y = df['type'].to_numpy(dtype=np.int64)
            return x, y
        except FileNotFoundError:
            print(f"Error: Cannot find the file at {filepath}")
            return None, None

    train_x, train_y = load_from_csv(TRAIN_FILE, dict1, db_id_to_name_map)
    val_x, val_y = load_from_csv(VALIDATION_FILE, dict1, db_id_to_name_map)
    test_x, test_y = load_from_csv(TEST_FILE, dict1, db_id_to_name_map)

    if train_x is None or val_x is None or test_x is None:
        print("Aborting due to missing data files.")
        return

    print(f"Data loaded: Train={len(train_x)}, Validation={len(val_x)}, Test={len(test_x)}")

    # 3. 初始化网络
    net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name_ids_list),
                        GNN2(dataset, tail_len, relation_len, args, dict1, drug_name_ids_list),
                        GNN3(dataset, tail_len, relation_len, args, dict1, drug_name_ids_list),
                        GNN4(dataset, tail_len, relation_len, args, dict1, drug_name_ids_list),
                        FusionLayer(args))

    # 4. 训练和验证
    best_model_state = train_and_validate(train_x, train_y, val_x, val_y, net)

    # 5. 最终测试
    if best_model_state:
        print("\nLoading best model for final evaluation on the test set...")
        net.load_state_dict(best_model_state)
    else:
        print("\nWarning: No best model found. Evaluating with the last model state.")

    net.eval()
    with torch.no_grad():
        test_x_tensor = torch.LongTensor(test_x)
        test_output_tensor = F.softmax(net(test_x_tensor), dim=1)

        test_loss = nn.CrossEntropyLoss()(test_output_tensor, torch.LongTensor(test_y)).item()
        y_pred = torch.argmax(test_output_tensor, dim=1).numpy()
        y_score = test_output_tensor.numpy()
        y_true = test_y

        result_all, _ = evaluate(y_pred, y_score, y_true, args.event_num)
        save_result("../result/", "all", result_all)

        print("\n--- Final Test Set Evaluation ---")
        print(f"Loss:      {test_loss:.4f}")
        print(f"Accuracy:  {result_all[0][0]:.4f}")
        print(f"F1-Score:  {result_all[6][0]:.4f} (Macro)")
        print(f"Recall:    {result_all[10][0]:.4f} (Macro)")
        print(f"Precision: {result_all[8][0]:.4f} (Macro)")
        print("---------------------------------")


if __name__ == '__main__':
    setup_seed()
    main()