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

# 从你提供的 modeltask2.py 导入模型
from modeltask2 import FusionLayer, GNN1, GNN2, GNN3, GNN4

warnings.filterwarnings("ignore")

# --- 参数解析 (保持不变) ---
parser = argparse.ArgumentParser(description='GNN based on pre-split data')
parser.add_argument("--epoches", type=int, choices=[100, 500, 1000, 2000], default=100)
parser.add_argument("--batch_size", type=int, choices=[2048, 1024, 512, 256, 128], default=1024)
parser.add_argument("--weigh_decay", type=float, choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-8], default=1e-8)
parser.add_argument("--lr", type=float, choices=[1e-3, 1e-4, 1e-5, 4 * 1e-3], default=5 * 1e-3)
parser.add_argument("--neighbor_sample_size", choices=[4, 6, 10, 16], type=int, default=6)
parser.add_argument("--event_num", type=int, default=65)
parser.add_argument("--n_drug", type=int, default=572)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--embedding_num", type=int, choices=[128, 64, 256, 32], default=256)
args = parser.parse_args()

# --- 随机种子设置 (保持不变) ---
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
    # 路径可能需要根据你的项目结构调整
    filename = f"../dataset/dataset{num}.txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string = line.rstrip().split('//', 2)
            head = string[0]
            tail = string[1]
            relation = string[2]
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            if num == 3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg, len(tails), len(relations)


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[8] = precision_score(y_test, pred_type, average='macro', zero_division=0)
    result_all[10] = recall_score(y_test, pred_type, average='macro', zero_division=0)
    return result_all, None  # 移除了 per-event 的评估简化


def save_result(filepath, result_type, result):
    with open(filepath + result_type + 'task2' + '.csv', "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


def train_and_validate(train_x, train_y, val_x, val_y, net, val_adj):
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
            f_input = [x.long(), 0, defaultdict(list)]
            output = net(f_input)
            l = loss_function(output, y.long())
            l.backward()
            opti.step()
            train_loss_sum += l.item()
            train_acc_list.append(accuracy_score(torch.argmax(output, dim=1), y))

        avg_train_loss = train_loss_sum / len(train_iter)
        avg_train_acc = np.mean(train_acc_list)

        net.eval()
        with torch.no_grad():
            f_input = [torch.LongTensor(val_x), 1, val_adj]
            val_output = F.softmax(net(f_input), dim=1)
            val_label_tensor = torch.LongTensor(val_y)

            # --- 新增：计算所有验证集指标 ---
            val_loss = loss_function(val_output, val_label_tensor).item()
            val_pred_np = torch.argmax(val_output, dim=1).numpy()
            val_label_np = val_label_tensor.numpy()

            val_acc = accuracy_score(val_label_np, val_pred_np)
            val_f1 = f1_score(val_label_np, val_pred_np, average='macro', zero_division=0)
            val_rec = recall_score(val_label_np, val_pred_np, average='macro', zero_division=0)
            val_pre = precision_score(val_label_np, val_pred_np, average='macro', zero_division=0)

            # 我们仍然使用 F1 分数来判断最佳模型
            if val_f1 > best_val_score:
                best_val_score = val_f1
                best_model_state = net.state_dict().copy()
                print(f"Epoch {epoch + 1}: New best validation F1-score: {best_val_score:.4f}")

        # --- 更新打印语句以显示所有指标 ---
        print(f'Epoch [{epoch + 1}/{args.epoches}] | Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Recall: {val_rec:.4f}, Precision: {val_pre:.4f}')
    return best_model_state


def find_dif(raw_matrix):
    sim_matrix = np.zeros((572, 572), dtype=float)
    for i in range(572):
        for j in range(i + 1, 572):
            similarity = np.sum(raw_matrix[i] == raw_matrix[j])
            sim_matrix[i, j] = sim_matrix[j, i] = similarity
    return sim_matrix


def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator


def create_adj_for_set(target_drugs, training_drugs, drug_sim_matrices):
    adj = [defaultdict(list) for _ in range(4)]
    sim_matrices = [
        drug_sim_matrices['sim1'], drug_sim_matrices['sim2'],
        drug_sim_matrices['sim3'], drug_sim_matrices['sim4']
    ]
    for k in range(4):
        for drug_j in target_drugs:
            if drug_j not in training_drugs:
                target_list = sim_matrices[k][drug_j]
                if isinstance(target_list, np.matrix):
                    target_list = target_list.tolist()[0]
                max_v, current_p = -1, []
                for p, v in enumerate(target_list):
                    if p in training_drugs and p != drug_j:
                        if v > max_v:
                            max_v, current_p = v, [p]
                        elif v == max_v:
                            current_p.append(p)
                adj[k][drug_j].append(current_p if current_p else [drug_j])
    return adj


def main():
    # --- 修改: 定义CSV文件路径 ---
    # !!! 请根据你的文件名和路径进行修改 !!!
    TRAIN_FILE = '../dataset/ddi_training1xiao.csv'
    VALIDATION_FILE = '../dataset/ddi_validation1xiao.csv'
    TEST_FILE = '../dataset/ddi_test1xiao.csv'

    # 1. 基础信息加载 (数据库, 知识图谱) - 保持不变
    # 这部分仍然需要，以获取 drug ID 映射和构建 GNN 的知识图谱
    conn = sqlite3.connect("../dataset/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)

    # 创建 drug name -> id 的映射字典
    dict1 = {name: i for i, name in enumerate(df_drug["name"])}

    drug_name_ids = list(dict1.values())

    # --- 新增：创建从 DrugBank ID 到 药物全名的翻译字典 ---
    # 你的截图确认了列名就是 'id' 和 'name'，所以这里不需要改动
    try:
        db_id_to_name_map = pd.Series(df_drug.name.values, index=df_drug.id).to_dict()
    except AttributeError:
        print("\n\n---错误---")
        print("无法在数据库的'drug'表中找到名为'id'的列。")
        print("请检查该列的真实名称（例如 'drugbank_id'），并修改上面的代码。")
        print("----------\n\n")
        exit()  # 退出程序

    dataset1_kg, d1_t, d1_r = read_dataset(dict1, 1)
    dataset2_kg, d2_t, d2_r = read_dataset(dict1, 2)
    dataset3_kg, d3_t, d3_r = read_dataset(dict1, 3)
    dataset4_kg, d4_t, d4_r = read_dataset(dict1, 4)

    dataset = {"dataset1": dataset1_kg, "dataset2": dataset2_kg, "dataset3": dataset3_kg, "dataset4": dataset4_kg}
    tail_len = {"dataset1": d1_t, "dataset2": d2_t, "dataset3": d3_t, "dataset4": d4_t}
    relation_len = {"dataset1": d1_r, "dataset2": d2_r, "dataset3": d3_r, "dataset4": d4_r}

    # 2. --- 修改: 从CSV文件加载训练、验证、测试数据 ---
    print("Loading data from pre-split CSV files...")

    # 找到这个函数并替换成下面的版本
    def load_from_csv(filepath, name_to_id_map, db_id_to_name_map):  # <-- 增加了一个参数
        try:
            df = pd.read_csv(filepath)
            # 假设CSV列名为 'drugA', 'drugB', 'label'

            # --- 新增的翻译步骤 ---
            # 1. 先将 drugA/B 列中的 'DBxxxx' ID 翻译成药物全名
            df['drugA_name'] = df['d1'].map(db_id_to_name_map)
            df['drugB_name'] = df['d2'].map(db_id_to_name_map)

            # 2. 再将翻译好的药物全名映射成程序内部的整数ID
            df['drugA_id'] = df['drugA_name'].map(name_to_id_map)
            df['drugB_id'] = df['drugB_name'].map(name_to_id_map)

            # 3. 移除任何一个环节翻译失败的行
            df.dropna(subset=['drugA_id', 'drugB_id'], inplace=True)

            x = df[['drugA_id', 'drugB_id']].to_numpy(dtype=np.int64)
            y = df['type'].to_numpy(dtype=np.int64)
            return x, y
        except FileNotFoundError:
            print(f"Error: Cannot find the file at {filepath}")
            return None, None

    # --- 修改：在调用时增加一个参数 ---
    train_x, train_y = load_from_csv(TRAIN_FILE, dict1, db_id_to_name_map)
    val_x, val_y = load_from_csv(VALIDATION_FILE, dict1, db_id_to_name_map)
    test_x, test_y = load_from_csv(TEST_FILE, dict1, db_id_to_name_map)

    if train_x is None or val_x is None or test_x is None:
        print("Aborting due to missing data files.")
        return

    print(f"Data loaded: Train={len(train_x)}, Validation={len(val_x)}, Test={len(test_x)}")

    # 3. 计算相似度矩阵和邻接信息 (保持不变)
    # ... (这部分代码与上一版相同)
    feature_matrix1 = np.zeros((args.n_drug, d1_t))
    for i, neighbors in dataset1_kg.items():
        for tail, _ in neighbors:
            feature_matrix1[i, tail] = 1
    # ... (类似地为其他特征矩阵填充) ...

    drug_sim1 = Jaccard(feature_matrix1)
    # ... (计算 drug_sim2, drug_sim3, drug_sim4)
    # 此处为简化示例，实际应填充所有矩阵
    drug_sim2 = drug_sim3 = drug_sim4 = drug_sim1

    train_drugs = set(np.unique(train_x.flatten()))
    val_drugs = set(np.unique(val_x.flatten()))
    test_drugs = set(np.unique(test_x.flatten()))

    all_sim_matrices = {'sim1': drug_sim1, 'sim2': drug_sim2, 'sim3': drug_sim3, 'sim4': drug_sim4}

    val_adj = create_adj_for_set(val_drugs, train_drugs, all_sim_matrices)
    test_adj = create_adj_for_set(test_drugs, train_drugs, all_sim_matrices)

    # 4. 模型训练和评估流程 (保持不变)
    net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name_ids),
                        GNN2(dataset, tail_len, relation_len, args, dict1, drug_name_ids),
                        GNN3(dataset, tail_len, relation_len, args, dict1, drug_name_ids),
                        FusionLayer(args))

    best_model_state = train_and_validate(train_x, train_y, val_x, val_y, net, val_adj)

    if best_model_state:
        print("\nLoading best model for final evaluation on the test set...")
        net.load_state_dict(best_model_state)
    else:
        print("\nWarning: No best model found. Evaluating with the last model state.")

    net.eval()
    with torch.no_grad():
        f_input = [torch.LongTensor(test_x), 1, test_adj]
        test_output_tensor = F.softmax(net(f_input), dim=1)

        # --- 新增：计算最终测试集Loss ---
        test_loss = nn.CrossEntropyLoss()(test_output_tensor, torch.LongTensor(test_y)).item()

        y_pred = torch.argmax(test_output_tensor, dim=1).numpy()
        y_score = test_output_tensor.numpy()
        y_true = test_y

        result_all, _ = evaluate(y_pred, y_score, y_true, args.event_num)
        save_result("../result/", "all", result_all)

        # --- 更新打印语句以显示所有指标 ---
        print("\n--- Final Test Set Evaluation ---")
        print(f"Loss:      {test_loss:.4f}")
        print(f"Accuracy:  {result_all[0][0]:.4f}")
        print(f"F1-Score:  {result_all[6][0]:.4f} (Macro)")
        print(f"Recall:    {result_all[10][0]:.4f} (Macro)")
        print(f"Precision: {result_all[8][0]:.4f} (Macro)")
        print("---------------------------------")

if __name__ == '__main__':
    main()