import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import time

def get_bm_scaffold(smiles):
    """提取分子的 Bemis-Murcko 骨架并返回其标准化 SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # 提取最大连通片段（处理脱盐操作）
        frags = Chem.GetMolFrags(mol, asMols=True)
        mol = max(frags, key=lambda x: x.GetNumAtoms())
        
        # 计算 BM 骨架
        core = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(core)
        
        # 如果分子是纯线性链（没有环），返回特定标识
        return scaffold_smi if scaffold_smi else "linear_chain"
    except Exception:
        return None

if __name__ == "__main__":
    # 1. 读取清洗好的 PARP2 数据
    input_file = '/share/home/u25511/wangyan/del数据/CK1a_HitGen_positive_data_阳性.csv'
    df = pd.read_csv(input_file)
    
    # 【核心修改】：仅提取阳性分子
    df_pos = df[df['label'] == 1].copy()
    print(f"🚀 开始处理，原始数据共 {len(df)} 个，成功提取 {len(df_pos)} 个 CAIX 阳性分子。")
    start_time = time.time()

    # 2. 计算每个阳性分子的 BM 骨架
    tqdm.pandas(desc="计算阳性分子 BM 骨架")
    df_pos['BM_Scaffold'] = df_pos['SMILES'].progress_apply(get_bm_scaffold)
    
    # 清理异常分子
    initial_len = len(df_pos)
    df_pos = df_pos.dropna(subset=['BM_Scaffold']).reset_index(drop=True)
    if len(df_pos) < initial_len:
        print(f"⚠️ 过滤了 {initial_len - len(df_pos)} 个无法解析骨架的异常分子。")

    # 3. 统计每个阳性骨架簇的丰度 (Scaffold Count)
    df_pos['Scaffold_Count'] = df_pos.groupby('BM_Scaffold')['BM_Scaffold'].transform('count')
    
    # 按照包含的活性分子数降序排列
    df_pos = df_pos.sort_values(by=['Scaffold_Count', 'BM_Scaffold'], ascending=[False, True]).reset_index(drop=True)

    # 4. 提取独特的骨架统计信息
    scaffold_summary = df_pos[['BM_Scaffold', 'Scaffold_Count']].drop_duplicates().reset_index(drop=True)
    print(f"🧩 共提取出 {len(scaffold_summary)} 种独特的阳性 BM 骨架。")
    print("\n🏆 排名前 5 的最大活性骨架簇：")
    print(scaffold_summary.head(5))

    # 5. 保存带有骨架标签的阳性数据
    # 修改了文件名，加上了 _pos_only 标识
    output_file = '/share/home/u25511/wangyan/del数据/CK1a_DEL_BM_scaffolds.csv'
    df_pos.to_csv(output_file, index=False)

    print(f"\n✅ 全部完成！耗时: {time.time() - start_time:.2f} 秒")
    print(f"📁 阳性分子骨架数据已保存至: {output_file}")