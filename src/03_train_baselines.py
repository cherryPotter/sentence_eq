#! /usr/bin/python 
############################################
#
# Create time: 2025-10-20 19:23:37
# version 1.1
#
############################################

"""
03_train_baselines.py
GAM与XGBoost(单调约束)训练 + SHAP解释
输入：test_llm_output.json (大模型提取的特征)
目标：sentence_months (判处刑期，月)
"""
import argparse, pandas as pd, numpy as np, pathlib, json, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pygam import LinearGAM, s, te
import xgboost as xgb
import shap, os

def load_json_data(json_path):
    """
    从JSON文件加载数据并转换为DataFrame
    JSON格式: {filename: {features}}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 将嵌套字典转换为DataFrame
    rows = []
    for filename, features in data.items():
        row = {"filename": filename}
        row.update(features)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} cases from {json_path}")
    return df

def build_design(df):
    """
    构建特征矩阵X和目标变量y
    使用统一量表评分和其他特征
    """
    df = df.copy()
    
    # 移除没有目标变量的样本
    df = df[df["sentence_months"].notna()].copy()
    print(f"Cases with valid sentence_months: {len(df)}")
    
    # 特征工程
    # 1. 统一量表评分（核心特征）
    df["harm_score"] = df["harm_score"].fillna(0)
    df["risk_score"] = df["risk_score"].fillna(0)
    df["mitigating_score"] = df["mitigating_score"].fillna(0)
    df["aggravating_score"] = df["aggravating_score"].fillna(0)
    
    # 2. 受贿金额（对数变换，处理偏态分布）
    df["log_amount"] = np.log1p(df["total_bribe_amount"].fillna(0))
    
    # 3. 从重情节数量
    aggravating_cols = [
        "prior_party_or_admin_discipline",
        "prior_intent_crime_record", 
        "used_for_illegal_activities",
        "refused_to_disclose_or_recover",
        "bad_impact_or_serious_consequence",
        "repeated_solicitation",
        "caused_public_loss",
        "sought_promotion_adjustment"
    ]
    df["aggravating_count"] = df[aggravating_cols].sum(axis=1)
    
    # 4. 被告人数量
    df["num_defendants"] = df["num_defendants"].fillna(1)
    
    # 选择特征
    feature_cols = [
        "harm_score",           # 社会危害性评分
        "risk_score",           # 人身危险性评分  
        "mitigating_score",     # 减轻情节评分
        "aggravating_score",    # 加重情节评分
        "log_amount",           # 受贿金额(对数)
        "aggravating_count",    # 从重情节数量
        "num_defendants"        # 被告人数量
    ]
    
    X = df[feature_cols]
    y = df["sentence_months"]
    
    print(f"\nFeature summary:")
    print(X.describe())
    print(f"\nTarget (sentence_months) summary:")
    print(y.describe())
    
    return X, y, df

def train_gam(X, y):
    """
    训练GAM模型，带单调约束
    特征顺序: harm_score, risk_score, mitigating_score, aggravating_score, 
              log_amount, aggravating_count, num_defendants
    单调性: +, +, -, +, +, +, +
    """
    # GAM with monotonic constraints
    # harm_score: 增加 (+)
    # risk_score: 增加 (+)
    # mitigating_score: 减少 (-)
    # aggravating_score: 增加 (+)
    # log_amount: 增加 (+)
    # aggravating_count: 增加 (+)
    # num_defendants: 增加 (+)
    
    gam = LinearGAM(
        s(0, constraints='monotonic_inc') +    # harm_score
        s(1, constraints='monotonic_inc') +    # risk_score
        s(2, constraints='monotonic_dec') +    # mitigating_score
        s(3, constraints='monotonic_inc') +    # aggravating_score
        s(4, constraints='monotonic_inc') +    # log_amount
        s(5, constraints='monotonic_inc') +    # aggravating_count
        s(6, constraints='monotonic_inc')      # num_defendants
    )
    
    # Grid search for best parameters
    gam = gam.gridsearch(X.values, y.values)
    
    print(f"GAM trained with {len(X)} samples")
    return gam

def train_xgb_monotone(X, y):
    """
    训练XGBoost模型，带单调约束
    单调约束对应特征顺序：
    (harm+, risk+, mitig-, aggr+, amount+, aggr_count+, num_def+)
    """
    # 单调约束: 1=增加, -1=减少, 0=无约束
    # 特征顺序: harm_score, risk_score, mitigating_score, aggravating_score, 
    #          log_amount, aggravating_count, num_defendants
    params = dict(
        max_depth=4, 
        eta=0.05,
        subsample=0.8, 
        colsample_bytree=0.8,
        objective="reg:squarederror",
        monotone_constraints="(1,1,-1,1,1,1,1)",  # 对应7个特征的单调性
        min_child_weight=3,
        gamma=0.1,
        base_score=y.mean()  # 明确设置base_score为训练集均值，避免SHAP兼容性问题
    )
    
    dtrain = xgb.DMatrix(X, label=y, feature_names=X.columns.tolist())
    bst = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    
    print(f"XGBoost trained with {len(X)} samples")
    return bst

def main():
    ap = argparse.ArgumentParser(description="训练GAM和XGBoost模型预测刑期")
    ap.add_argument("--in", dest="inp", required=True, help="输入JSON文件路径（如 test_llm_output.json）")
    ap.add_argument("--out", required=True, help="输出目录路径")
    args = ap.parse_args()
    
    print("="*80)
    print("训练量刑预测模型（GAM + XGBoost）")
    print("="*80)
    
    # 1. 加载数据
    print("\n[1/5] Loading data...")
    df = load_json_data(args.inp)
    
    # 2. 构建特征
    print("\n[2/5] Building features...")
    X, y, df_processed = build_design(df)
    
    # 3. 划分训练集和测试集
    print(f"\n[3/5] Splitting data (80% train, 20% test)...")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {len(Xtr)} samples")
    print(f"Test set: {len(Xte)} samples")
    
    # 4. 训练模型
    print("\n[4/5] Training models...")
    print("\n  [4a] Training GAM...")
    gam = train_gam(Xtr, ytr)
    yhat_gam = gam.predict(Xte.values)
    
    print("\n  [4b] Training XGBoost with monotonic constraints...")
    bst = train_xgb_monotone(Xtr, ytr)
    dtest = xgb.DMatrix(Xte, feature_names=X.columns.tolist())
    yhat_xgb = bst.predict(dtest)
    
    # 5. 评估模型
    print("\n[5/5] Evaluating models...")
    
    # GAM metrics
    mae_gam = mean_absolute_error(yte, yhat_gam)
    rmse_gam = np.sqrt(mean_squared_error(yte, yhat_gam))
    r2_gam = r2_score(yte, yhat_gam)
    
    # XGBoost metrics
    mae_xgb = mean_absolute_error(yte, yhat_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(yte, yhat_xgb))
    r2_xgb = r2_score(yte, yhat_xgb)
    
    print("\nModel Performance:")
    print(f"  GAM:")
    print(f"    MAE:  {mae_gam:.2f} months ({mae_gam/12:.2f} years)")
    print(f"    RMSE: {rmse_gam:.2f} months ({rmse_gam/12:.2f} years)")
    print(f"    R²:   {r2_gam:.4f}")
    print(f"\n  XGBoost:")
    print(f"    MAE:  {mae_xgb:.2f} months ({mae_xgb/12:.2f} years)")
    print(f"    RMSE: {rmse_xgb:.2f} months ({rmse_xgb/12:.2f} years)")
    print(f"    R²:   {r2_xgb:.4f}")
    
    # 6. 保存结果
    print(f"\nSaving results to {args.out}...")
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    
    # Save models
    # GAM模型使用pickle保存
    with open(f"{args.out}/gam_model.pkl", 'wb') as f:
        pickle.dump(gam, f)
    
    # XGBoost模型使用原生方法保存
    bst.save_model(f"{args.out}/xgb_monotone.json")
    
    # Save metrics
    metrics = {
        "GAM": {
            "MAE_months": float(mae_gam),
            "MAE_years": float(mae_gam/12),
            "RMSE_months": float(rmse_gam),
            "RMSE_years": float(rmse_gam/12),
            "R2": float(r2_gam)
        },
        "XGBoost": {
            "MAE_months": float(mae_xgb),
            "MAE_years": float(mae_xgb/12),
            "RMSE_months": float(rmse_xgb),
            "RMSE_years": float(rmse_xgb/12),
            "R2": float(r2_xgb)
        },
        "feature_names": X.columns.tolist()
    }
    
    with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        "actual": yte.values,
        "pred_gam": yhat_gam,
        "pred_xgb": yhat_xgb,
        "error_gam": yte.values - yhat_gam,
        "error_xgb": yte.values - yhat_xgb
    })
    predictions_df.to_csv(f"{args.out}/predictions.csv", index=False)
    
    # SHAP analysis (for XGBoost)
    print("\nCalculating SHAP values for XGBoost...")
    try:
        # 使用model_output参数来避免base_score的兼容性问题
        explainer = shap.TreeExplainer(bst, model_output='raw')
        shap_values = explainer.shap_values(Xte)
        
        # Save SHAP values
        np.save(f"{args.out}/shap_values.npy", shap_values)
        Xte.to_csv(f"{args.out}/Xte.csv", index=False)
        print("  ✓ SHAP values calculated and saved")
    except Exception as e:
        print(f"  ⚠ Warning: Could not calculate SHAP values: {e}")
        print("  Skipping SHAP analysis (models are still saved)")
        # Save test set anyway for manual analysis
        Xte.to_csv(f"{args.out}/Xte.csv", index=False)
    
    # Feature importance (XGBoost)
    importance = bst.get_score(importance_type='gain')
    importance_df = pd.DataFrame([
        {"feature": k, "importance": v} 
        for k, v in importance.items()
    ]).sort_values("importance", ascending=False)
    importance_df.to_csv(f"{args.out}/feature_importance.csv", index=False)
    
    print("\n" + "="*80)
    print("✓ Training completed successfully!")
    print(f"  Models saved to: {args.out}/")
    print(f"  - gam_model.pkl")
    print(f"  - xgb_monotone.json")
    print(f"  - metrics.json")
    print(f"  - predictions.csv")
    print(f"  - shap_values.npy")
    print(f"  - feature_importance.csv")
    print("="*80)

if __name__ == "__main__":
    main()
