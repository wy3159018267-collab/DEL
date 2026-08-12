import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm

# ==========================================
# 1. 加载数据 (请替换为你真实的 CSV 路径)
# 假设你的 CSV 文件里有一列叫 'BM_Scaffold'
# ==========================================
del_file = '/share/home/u25511/wangyan/del数据/sEH_DEL_BM_scaffolds.csv'      
chembl_file = '/share/home/u25511/wangyan/chembl数据/sEH_BM_scaffolds_pos_only.csv' 

df_del = pd.read_csv(del_file)
df_chembl = pd.read_csv(chembl_file)

# 提取骨架列表 (去重防万一)
del_smiles = df_del['BM_Scaffold'].dropna().unique().tolist()
chembl_smiles = df_chembl['BM_Scaffold'].dropna().unique().tolist()

print(f"载入骨架: DEL={len(del_smiles)}, ChEMBL={len(chembl_smiles)}")

# ==========================================
# 2. 计算 Morgan 指纹 (Radius=2, 2048 bits)
# ==========================================
def get_fingerprints(smiles_list):
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_smiles.append(smi)
    return fps, valid_smiles

print("计算 ChEMBL 骨架指纹...")
chembl_fps, valid_chembl_smiles = get_fingerprints(chembl_smiles)

print("计算 DEL 骨架指纹...")
del_fps, valid_del_smiles = get_fingerprints(del_smiles)

# ==========================================
# 3. 计算最近邻最大相似度 (Nearest Neighbor)
# ==========================================
max_similarities = []
closest_chembl_smiles = []

print("正在计算跨库相似度...")
for del_fp in tqdm(del_fps):
    # 批量计算 1 个 DEL 指纹 对 所有 ChEMBL 指纹的相似度
    sims = DataStructs.BulkTanimotoSimilarity(del_fp, chembl_fps)
    
    # 找到最大值及其对应的 ChEMBL 骨架
    max_sim = max(sims)
    max_idx = sims.index(max_sim)
    
    max_similarities.append(max_sim)
    closest_chembl_smiles.append(valid_chembl_smiles[max_idx])

# 将结果存入新的 DataFrame
result_df = pd.DataFrame({
    'DEL_Scaffold': valid_del_smiles,
    'Max_Tanimoto_to_ChEMBL': max_similarities,
    'Closest_ChEMBL_Scaffold': closest_chembl_smiles
})

# 按相似度从小到大排序（排在前面的是最新颖的骨架）
result_df = result_df.sort_values(by='Max_Tanimoto_to_ChEMBL')
result_df.to_csv('sEH_DEL_vs_ChEMBL_Analysis.csv', index=False)
print("计算完成，结果已保存至 DEL_vs_ChEMBL_Analysis.csv")

# ==========================================
# 4. 数据可视化：相似度分布图
# ==========================================
plt.style.use('default')
plt.figure(figsize=(10, 6))

# 绘制 KDE 密度分布和直方图
sns.histplot(result_df['Max_Tanimoto_to_ChEMBL'], bins=30, kde=True, color='#4C72B0', edgecolor='black')

# 画一条 0.4 的阈值虚线 (通常 Tanimoto < 0.4 被认为具有极高新颖性)
# plt.axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='Novelty Threshold (0.4)')

plt.title('Novelty Analysis: DEL vs ChEMBL Scaffolds (sEH)', fontsize=16, fontweight='bold')
plt.xlabel('Maximum Tanimoto Similarity to ChEMBL', fontsize=14)
plt.ylabel('Count of DEL Scaffolds', fontsize=14)
plt.legend()
plt.tight_layout()

plt.savefig('sEH_DEL_Distribution.png', dpi=300)
print("✅ 相似度分布图已保存至 sEH_DEL_Distribution.png")