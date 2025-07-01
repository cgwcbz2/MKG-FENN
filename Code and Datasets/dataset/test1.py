import numpy as np

# 将 'your_file.npy' 替换成你的文件名
file_path = './drug_cell_gpt_all/drug_embed_all.npy'

# 加载 .npy 文件
data = np.load(file_path)


print(data)
print(data.shape)
# import numpy as np
# import pandas as pd

# # 读取 motif 数据
# motif_data = np.load("drug_substructure_dict.npz", allow_pickle=True)
# 
# # 读取 drug 表（假设含有 'id' 和 'name' 两列）
# df = pd.read_csv("./exported_tables/drug.csv")  # 列如 id: DB00977, name: Glucosamine
# 
# # 输出文件
# with open("motif_dataset2.txt", "w") as f:
#     for _, row in df.iterrows():
#         dbid = row['id']       # 如 DB00977
#         drug_name = row['name']  # 如 Glucosamine
# 
#         if dbid in motif_data:
#             motifs = motif_data[dbid].tolist()
#             for motif in motifs:
#                 f.write(f"{drug_name}//{motif}//include\n")
#         else:
#             print(f"⚠️ {dbid} 不在 motif_data 中，跳过")

# import numpy as np
#
# # 加载 .npz 文件
# data = np.load('drug_substructure_dict.npz', allow_pickle=True)
#
# # 构建 motif_map：一个标准的 dict
# motif_map = {}
# for drug_id in data.files:
#     motif_map[drug_id] = data[drug_id].tolist()  # 转成 Python list 更安全
# with open('motif_dataset.txt', 'w') as f:
#     for drug_id, motifs in motif_map.items():
#         for motif_id in motifs:
#             f.write(f"{drug_id}//{motif_id}//include\n")
