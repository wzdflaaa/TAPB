import argparse
import json
import pathlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#命令行解析器
def parse_args():
    #命令行参数解析器
    parser = argparse.ArgumentParser(description="Analyse dataset prior bias at global/target/drug level")
    #数据集名称参数
    parser.add_argument('--data', required=True, type=str, 
                        help='dataset name under ./datasets/')
    #数据集划分方式参数
    parser.add_argument('--split', default='random', type=str, help="split name under dataset directory",
                        choices=['random', 'cold', 'cluster'])
    #输出结果保存路径参数
    parser.add_argument('--output_dir', default='results/prior_bias_analysis', type=str,
                        help='directory to save the analysis results')
    #置换实验参数
    parser.add_argument("--num_perm",default=500,type=int,
                        help="number of permutations for signficance test")
    return parser.parse_args()

#自动收集当前split下训练集文件
def collect_csv_files(dataset_dir:pathlib.Path):
    candidates=["train_with_id.csv",
                "source_train_with_id.csv",
                "source_with_id.csv"]
    csv_files=[dataset_dir/name for name in candidates if(dataset_dir/name).exists()]   
    if not csv_files:
        raise FileNotFoundError(
            f"Cannot find train/source_train file in {dataset_dir}. "
            f"Available files: {[p.name for p in dataset_dir.glob('*_with_id.csv')]}"
        )
    return sorted(csv_files)

#从数据中提取pr 列名
def pick_protein_column(df:pd.DataFrame):
    for col in ["pr_id","target_id","Target","Protein"]:
        if col in df.columns:
            return col  
    raise KeyError("cannot find protein column in the dataset, try:['pr_id','target_id','Target','Protein']")

#从数据中提取drug_id(drug列名)
def pick_drug_column(df:pd.DataFrame):
    for col in ["drug_id","Drug","SMILES"]:
        if col in df.columns:
            return col  
    raise KeyError("cannot find drug column in the dataset, try:['drug_id','Drug','SMILES']")

# 清洗标签列，转换为二元分类标签（0/1）-数值
def sanitize_labels(df:pd.DataFrame,label_col:str="Y"):
    if label_col not in df.columns:
        raise KeyError(f"Label colum '{label_col}' not found in  dataset")
    df=df.copy()
    df[label_col]=pd.to_numeric(df[label_col],errors="coerce").fillna(0).astype(int)
    df[label_col]=(df[label_col]>0).astype(int)
    return df

#计算某侧分组维度(pr/drug) 统计信息
def compute_group_metrics(df:pd.DataFrame,group_col:str,label_col:str):
    """
    计算每个 group 的:
      - sample
      - positive
      - negative
      - p
      - abs_gap = |p - p_global|
      - z = n * |p - p_global| / N
    仅返回加权整体偏差强度 Zw
    """
    #全局样本量统计
    N=len(df) #总样本数量
    p_global=df[label_col].sum()/N #全局正样本比例
    
    
    grouped=df.groupby(group_col)[label_col]  #按group_col分组
    n=grouped.count() #每个组的样本数量
    
    #计算每个组的统计信息：正样本数量 负样本数量 正样本比例
    positive=grouped.sum() #每个组的正样本数量
    negative=n-positive #每个组的负样本数量
    
    #每个组的正样本比例
    p=positive/n 
    #计算偏差强度
    abs_gap=(p-p_global).abs() #每个组的偏差绝对值
    z=(n*abs_gap)/N
    
    result = pd.DataFrame({
        group_col: n.index.astype(str),
        "sample": n.values,
        "positive": positive.values,
        "negative": negative.values,
        "p": p.values,
        "abs_gap": abs_gap.values,
        "z": z.values,
    }).sort_values(by="z", ascending=False).reset_index(drop=True) #按照 z 值降序排序

    summary = {
        "group_col": group_col,
        "num_groups": int(len(result)),
        "p_global": float(p_global),
        "Zw_weighted": float(result["z"].sum()),
    } #统计结果摘要--所有pr或drug的加权偏差强度总和 Zw（越大说明整体偏差越严重）
    return result, summary

#置换检验
def permutation_test_group_prior(df:pd.DataFrame,group_col:str,label_col:str,num_perm:int=500,random_seed:int=42):
    
    rng=np.random.default_rng(random_seed)
    
    _,observed_summary,  = compute_group_metrics(df, group_col, label_col)
    observed_Zw = observed_summary["Zw_weighted"]
    
    permuted_values = []
    y=df[label_col].to_numpy().copy()#标签数组
    
    for _ in range(num_perm):
        y_permuted = rng.permutation(y) #标签置换
        tmp_df=df[[group_col]].copy() #只保留group_col列的临时数据框
        tmp_df[label_col]=y_permuted 
        _, permuted_summary = compute_group_metrics(tmp_df, group_col, label_col)
        permuted_Zw = permuted_summary["Zw_weighted"]
        permuted_values.append(permuted_Zw)
    
    p_value = (np.sum(np.array(permuted_values) >= observed_Zw) + 1) / (num_perm + 1)
    return {"observed_Zw": float(observed_Zw),
        "perm_mean_Zw": float(np.mean(permuted_values)),
        "perm_std_Zw": float(np.std(permuted_values)),
        "p_value": float(p_value),},permuted_values

# 绘制 target 与 drug 的 p/Z 直方图
def plot_histograms(pr_df: pd.DataFrame, drug_df: pd.DataFrame, output_png: pathlib.Path):
    # 创建 2x2 子图画布.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 绘制 target 的 p 分布
    axes[0, 0].hist(pr_df["p"], bins=30)
    axes[0, 0].set_title("protein p distribution")
    axes[0, 0].set_xlabel("p = positive / samples")
    axes[0, 0].set_ylabel("count")

    # 绘制 target 的 Z 分布
    axes[0, 1].hist(pr_df["z"], bins=30)
    axes[0, 1].set_title("protein Z distribution")
    axes[0, 1].set_xlabel("z = (n * |p - p_global|) / N")
    axes[0, 1].set_ylabel("count")

    # 绘制 drug 的 p 分布
    axes[1, 0].hist(drug_df["p"], bins=30)
    axes[1, 0].set_title("Drug p distribution")
    axes[1, 0].set_xlabel("p = positive / samples")
    axes[1, 0].set_ylabel("count")

    # 绘制 drug 的 Z 分布
    axes[1, 1].hist(drug_df["z"], bins=30)
    axes[1, 1].set_title("Drug Z distribution")
    axes[1, 1].set_xlabel("z = (n * |p - p_global|) / N")
    axes[1, 1].set_ylabel("count")

    # 自动调整子图间距避免标签重叠
    plt.tight_layout()
    # 保存图像到文件
    fig.savefig(output_png, dpi=200)
    # 释放图像资源
    plt.close(fig)

#绘制置换检验结果的分布图
def plot_permutation_distribution(permuted_values:list, observed_value:float, title:str, output_png:pathlib.Path):
    plt.figure(figsize=(8, 6))
    plt.hist(permuted_values, bins=30, alpha=0.7, label="Permuted Zw")
    plt.axvline(observed_value, color='red', linestyle='--', linewidth=2, label=f"Observed Zw = {observed_value:.4f}")
    plt.title(title)
    plt.xlabel("Zw (Weighted Bias Strength)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    
#获取对总偏差贡献最大的前k个实体
def get_shortcut_entities(group_stats:pd.DataFrame, group_col:str, top_k:int=10):
    top_df=group_stats.nlargest(top_k, 'z').copy().reset_index(drop=True) #按照 z 值降序选取前 k 个实体
    total_z=float(group_stats['z'].sum()) #计算所有实体的 z 值总和
    top_df['cum_z']=top_df['z'].cumsum() #计算累计z值(累计偏差综合)
    top_df['cum_contrib_ratio']=top_df['cum_z']/total_z if total_z > 0 else 0.0  #计算累计偏差强度占比--贡献率
    return top_df[[group_col, "sample", "positive", "negative", "p", "z", "cum_z", "cum_contrib_ratio"]] 
#绘制前k个实体的贡献率曲线图
def plot_topk_cumulative_curve(top_df:pd.DataFrame, group_name:str,output_png:pathlib.Path):
    plt.figure(figsize=(10, 6))
    x=np.arange(1,len(top_df)+1) #x轴为前k个实体的排名
    y=top_df['cum_contrib_ratio'] #y轴为累计贡献率
    # use plt.xticks to set tick values
    plt.xticks(x, rotation=45) #设置x轴刻度为整数并旋转标签
    plt.ylim(0,1.05)#设置y轴范围为0-1.05
    plt.plot(x, y, marker='o') #折线图
    plt.title(f"Top {len(top_df)} Entities: Cumulative z Contribution")
    plt.xlabel(f" {group_name} Top-k shortcut entities")
    plt.ylabel("Cumulative Contribution Ratio")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close() 

#保存分析结果
def save_group_analysis(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    out_dir: pathlib.Path,
    prefix: str,
    num_perm: int,
):
    """保存单组的分析结果"""
    group_stats, group_summary = compute_group_metrics(df, group_col, label_col)  # 计算组指标
    perm_summary, perm_values = permutation_test_group_prior(  # 置换检验
        df, group_col, label_col, num_perm=num_perm
    )

    # 定义输出文件路径
    stats_csv = out_dir / f"{prefix}_bias_stats.csv"
    perm_png = out_dir / f"{prefix}_permutation_test.png"
    top10_csv = out_dir / f"{prefix}_top10_shortcut_entities.csv"
    top10_curve_png = out_dir / f"{prefix}_top10_cumulative_contribution.png"

    # 保存结果
    group_stats.to_csv(stats_csv, index=False)  # 保存详细统计
    plot_permutation_distribution(  # 绘制置换检验图
        perm_values,
        perm_summary["observed_Zw"],
        f"{prefix.capitalize()} permutation test",
        perm_png
    )

    top10_df = get_shortcut_entities(group_stats, group_col, top_k=10)  # 获取前10个实体
    top10_df.to_csv(top10_csv, index=False)  # 保存前10个实体
    plot_topk_cumulative_curve(top10_df, prefix.capitalize(), top10_curve_png)  # 绘制累积贡献曲线

    merged_summary = {  # 合并汇总信息
        **group_summary,
        **perm_summary,
        "stats_csv": str(stats_csv),
        "perm_png": str(perm_png),
        "top10_csv": str(top10_csv),
        "top10_curve_png": str(top10_curve_png),
    }
    return group_stats, merged_summary  # 返回详细统计和汇总

def main():
    args = parse_args()
    # 构造数据目录路径：datasets/{data}/{split}
    dataset_dir = pathlib.Path("datasets") / args.data / args.split
    # 若目录不存在则报错
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    # 收集该目录下所有 with_id CSV 文件
    csv_files = collect_csv_files(dataset_dir)
    # 读取并合并所有 CSV（如 train/val/test，或 cluster 的 source/target）
    dataframes = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dataframes, ignore_index=True)
    # 检查标签列是否存在
    if "Y" not in df.columns:
        raise KeyError("Label column 'Y' not found in dataset")
    
    # 标签转为数值并限制为 0/1（兼容部分脏值）
    df = sanitize_labels(df, label_col="Y")
    
    protein_col = pick_protein_column(df)
    drug_col = pick_drug_column(df)
    
    #protein_stats = compute_group_metrics(df, protein_col, "Y")
    #drug_stats = compute_group_metrics(df, drug_col, "Y")
    p_global = df["Y"].mean()

    # 构造输出目录：{output_dir}/{data}/{split}
    out_dir = pathlib.Path(args.output_dir) / args.data / args.split
    # 递归创建输出目录
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #保存分析结果并绘制图表
    protein_stats, protein_summary = save_group_analysis(  # 分析蛋白质偏差
        df, protein_col, "Y", out_dir, "protein", args.num_perm
    )
    drug_stats, drug_summary = save_group_analysis(  # 分析药物偏差
        df, drug_col, "Y", out_dir, "drug", args.num_perm
    )

    combined_png = out_dir / "prior_bias_histograms.png"  # 联合直方图路径
    plot_histograms(protein_stats, drug_stats, combined_png)  # 绘制联合直方图

    summary = {  # 构建总体摘要
        "loaded_files": [str(f) for f in csv_files],
        "num_samples": int(len(df)),
        "global_positive_ratio": float(df["Y"].mean()),
        "protein_analysis": protein_summary,
        "drug_analysis": drug_summary,
        "combined_hist_png": str(combined_png),
    }

    summary_json = out_dir / "summary.json"  # 摘要文件路径
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)  # 保存摘要为JSON

    # 打印结果摘要 加载的文件列表 样本数量 全局正样本比例 以及蛋白质和药物的偏差强度 Zw 和 p 值 摘要的文件路径 联合直方图的文件路径
    print("=" * 80)
    print(f"Loaded files: {[str(f) for f in csv_files]}")
    print(f"Total train samples N: {len(df)}") 
    print(f"Global prior p_global: {df['Y'].mean():.6f}") 
    print("-" * 80)
    print(f"Protein Zw: {protein_summary['Zw_weighted']:.6f}, " 
          f"p-value: {protein_summary['p_value']:.6f}") 
    print(f"Drug    Zw: {drug_summary['Zw_weighted']:.6f}, " 
          f"p-value: {drug_summary['p_value']:.6f}") 
    print("-" * 80)
    print(f"Summary saved to: {summary_json}") 
    print(f"Combined histogram saved to: {combined_png}")  
    print("=" * 80)


    
if __name__ == "__main__":
    main()