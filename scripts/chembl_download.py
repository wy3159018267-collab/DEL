import pandas as pd
from chembl_webresource_client.new_client import new_client

def get_MAPK14_data_for_training():
    print("🚀 正在连接 ChEMBL 数据库，准备下载 MAPK14 (CHEMBL260) 数据...")
    
    target_id = 'CHEMBL260' # MAPK14
    
    # 1. 获取数据：只选 IC50, Ki, Kd (排除 EC50)
    activity = new_client.activity
    res = activity.filter(
        target_chembl_id=target_id, 
        standard_type__in=['IC50', 'Ki', 'Kd'], 
        target_organism="Homo sapiens"
    )
    
    # 2. 转换为 DataFrame
    df = pd.DataFrame.from_dict(res)
    
    if not df.empty:
        # 筛选核心列
        cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_type', 
                'standard_value', 'standard_units', 'pchembl_value']
        
        # 初步清洗：保留必要列，去除空值
        df = df[df.columns.intersection(cols)].dropna(subset=['canonical_smiles', 'standard_value'])
        
        # 确保数值列是数字格式
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df.dropna(subset=['standard_value'])

        # ---------------------------------------------------------
        # 3. 核心逻辑：Ki/Kd -> IC50 转换 (Value * 2)
        # ---------------------------------------------------------
        def convert_to_ic50(row):
            val = row['standard_value']
            dtype = row['standard_type']
            
            # 假设 standard_units 均为 nM (ChEMBL 默认)
            if dtype == 'IC50':
                return val
            elif dtype in ['Ki', 'Kd']:
                return val * 2.0  # 应用 Cheng-Prusoff 近似
            return val

        print("🔄 正在执行单位归一化 (Ki/Kd * 2 -> IC50)...")
        df['unified_IC50_nM'] = df.apply(convert_to_ic50, axis=1)
        
        # 标记数据来源类型，方便后续分析
        df['original_type'] = df['standard_type']
        
        # 最终只保留需要的列
        final_cols = ['molecule_chembl_id', 'canonical_smiles', 'unified_IC50_nM', 'original_type']
        df_final = df[final_cols]

        # 保存为 CSV
        output_file = "MAPK14_train_data_combined.csv"
        df_final.to_csv(output_file, index=False)
        
        print(f"✅ 成功！共处理 {len(df_final)} 条数据。")
        print(f"📁 文件已保存为: {output_file}")
        print("-" * 30)
        print("数据构成:")
        print(df['original_type'].value_counts())
        
    else:
        print("❌ 未找到数据，请检查网络连接。")

if __name__ == "__main__":
    get_MAPK14_data_for_training()
