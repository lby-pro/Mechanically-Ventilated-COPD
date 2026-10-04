#################### 导入需要的python库 ####################
## run start ##
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
from scipy.stats import norm
import numpy as np
import pandas as pd
from tableone import TableOne
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免tkinter相关错误
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import shap
## run end ##

####################### 1.数据导入 #######################
os.chdir(r"D:\桌面\桌面文件\储备课题\Lab-FI与机械通气COPD\机器学习内容\python机器学习课程FI-Lab机械通气COPD院内死亡")
data = pd.read_csv("数据/mimic_data_imputed.csv", encoding="GBK")
print(data.head())
print(data.info())


######################## 5.使用温莎方法处理异常值 #####################
print(data.info())

def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers), lower_bound, upper_bound

def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    lower_val = series.quantile(lower_percentile)
    upper_val = series.quantile(upper_percentile)
    return series.clip(lower=lower_val, upper=upper_val)

print(data.info())
continuous_vars = ["age","sofa","gcs","oasis","weight"]

for var in continuous_vars:
    outlier_count, lower_bound, upper_bound = detect_outliers(data[var])
    print(f"{var}：异常值数量={outlier_count}，下界={lower_bound:.2f}，上界={upper_bound:.2f}")
    if outlier_count > 0:
        original_stats = data[var].describe()
        data[var] = winsorize_series(data[var])
        new_stats = data[var].describe()
        print(f"\n{var}处理前统计描述:")
        print(original_stats[['min', '25%', '50%', '75%', 'max']])
        print(f"\n{var}处理后统计描述:")
        print(new_stats[['min', '25%', '50%', '75%', 'max']])
    else:
        print(f"{var}未检测到异常值，无需处理")

print(data.info())


##################### 7.数据集拆分 ######################

print(data.columns.tolist())
keep_vars = [col for col in data.columns if col not in ['subject_id']]
print(keep_vars)
data = data[keep_vars]
print(data.info())
train_data, test_data = train_test_split(data, test_size=0.3,
                                         stratify=data["hosp_expire"], random_state=2025)
train_data.to_csv("数据/train_data_notscaled.csv")
test_data.to_csv("数据/test_data_notscaled.csv")
print(train_data)

################ 9.连续型变量标准化#################
continuous_vars = ["age","sofa","gcs","oasis","weight"]
train_data[continuous_vars] = StandardScaler().fit_transform(train_data[continuous_vars])
train_data
train_data.to_csv("数据/train_data_scaled.csv")
test_data = test_data[keep_vars]
test_data[continuous_vars] = StandardScaler().fit_transform(test_data[continuous_vars])
test_data.to_csv("数据/test_data_scaled.csv")

###########10.使用R语言进行lasso回归及boruta算法，确认显著变量 #######
#经lasso回归筛选后重要变量
# 定义目标变量名称
target_var = 'hosp_expire'
#定义显著变量
significant_vars=["cerebrovascular","CRRT","sofa","oasis","fi_lab_score" ]
with open('生成的文件/significant_vars.pkl', 'wb') as f: # 将列名保存为文件，方便之后读取
    pickle.dump(significant_vars, f)

###加载上面导出的四个数据集 
train_data_notscaled = pd.read_csv("数据/train_data_notscaled.csv", encoding="GBK", index_col=0) # 训练集数据
test_data_notscaled = pd.read_csv("数据/test_data_notscaled.csv", encoding="GBK", index_col=0) # 测试集数据
train_data_scaled = pd.read_csv("数据/train_data_scaled.csv", encoding="GBK", index_col=0) # 训练集标准化数据
test_data_scaled = pd.read_csv("数据/test_data_scaled.csv", encoding="GBK", index_col=0) # 测试集标准化数据
#根据significant_vars及结局变量hosp_expire，构建四个新的数据框,用于后续机器学习训练
train_data_notscaled_sig = train_data_notscaled[significant_vars + [target_var]]
test_data_notscaled_sig = test_data_notscaled[significant_vars + [target_var]]
train_data_scaled_sig = train_data_scaled[significant_vars + [target_var]]
test_data_scaled_sig = test_data_scaled[significant_vars + [target_var]]
#打印新的四个数据框数据
print(train_data_notscaled_sig.head())
print(test_data_notscaled_sig.head())
print(train_data_scaled_sig.head())
print(test_data_scaled_sig.head())

###定义模型训练需要用到的自变量和结局变量###
# 逻辑回归模型不涉及超参数调优（使用的是非标准化数据集） 
X_train_logist = train_data_notscaled_sig[significant_vars] # 提取训练集中显著的自变量
X_train_logist_const = sm.add_constant(X_train_logist) # 训练集添加常数项（截距）
y_train_logist = train_data_notscaled_sig[target_var] # 结局变量
#剩余8大模型使用标准化数据集
X_train = train_data_scaled_sig.loc[:, train_data_scaled_sig.columns != target_var]
y_train = train_data_scaled_sig[target_var]

##定义模型测试集需要用到的自变量和结局##
# 得到测试集数据（Logistic模型）
X_test_logist = test_data_notscaled_sig[significant_vars]
X_test_logist_const = sm.add_constant(X_test_logist)
y_test_logist = test_data_notscaled_sig[target_var]
# 得到测试数据集（机器学习模型）
X_test = test_data_scaled_sig.loc[:, test_data_scaled_sig.columns != target_var]
y_test = test_data_scaled_sig[target_var] 

##################11.构建预测模型######################
############# 11.0.模型选择配置 ########
# ================================================================
# 【重要】在这里选择要训练和评估的模型
# 设置为 True 表示启用该模型，设置为 False 表示禁用该模型
# ================================================================
# 模型选择配置 - True=启用, False=禁用
# 示例1: 只训练5个模型 (Logistic, RF, XGB, LGB, ANN)
MODEL_SELECTION = {
    'logistic': True,   # Logistic回归模型
    'tree': True,       # 决策树模型
    'rf': True,         # 随机森林模型
    'xgb': True,        # XGBoost模型
    'lgb': False,        # LightGBM模型
    'svm': False,        # 支持向量机模型
    'adaboost': True,   # AdaBoost模型
    'knn': True,        # K近邻模型
    'ann': True         # 人工神经网络模型
}

# 根据选择自动生成启用的模型列表
SELECTED_MODELS = [k for k, v in MODEL_SELECTION.items() if v]
print("="*60)
print("📌 当前选中的模型:")
for i, model_name in enumerate(SELECTED_MODELS, 1):
    print(f"   {i}. {model_name.upper()}")
print(f"   共 {len(SELECTED_MODELS)} 个模型")
print("="*60)

# 定义模型的完整配置信息（用于后续代码的动态调用）
MODEL_CONFIG = {
    'logistic': {
        'name': 'Logistic',
        'file': 'logistic_model.pkl',
        'display_name': 'Logistic',
        'use_scaled_data': False,
        'predict_method': 'predict',
        'has_hyperparameter_tuning': False
    },
    'tree': {
        'name': 'Decision Tree',
        'file': 'tree_model.pkl',
        'display_name': 'tree',
        'use_scaled_data': True,
        'predict_method': 'predict_proba',
        'has_hyperparameter_tuning': True
    },
    'rf': {
        'name': 'Random Forest',
        'file': 'rf_model.pkl',
        'display_name': 'rf',
        'use_scaled_data': True,
        'predict_method': 'predict_proba',
        'has_hyperparameter_tuning': True
    },
    'xgb': {
        'name': 'XGBoost',
        'file': 'xgb_model.pkl',
        'display_name': 'XGBoost',
        'use_scaled_data': True,
        'predict_method': 'predict_proba',
        'has_hyperparameter_tuning': True
    },
    'lgb': {
        'name': 'LightGBM',
        'file': 'lgb_model.pkl',
        'display_name': 'LightGBM',
        'use_scaled_data': True,
        'predict_method': 'predict_proba',
        'has_hyperparameter_tuning': True
    },
    'svm': {
        'name': 'SVM',
        'file': 'svm_model.pkl',
        'display_name': 'SVM',
        'use_scaled_data': True,
        'predict_method': 'predict_proba',
        'has_hyperparameter_tuning': True
    },
    'adaboost': {
        'name': 'AdaBoost',
        'file': 'adaboost_model.pkl',
        'display_name': 'AdaBoost',
        'use_scaled_data': True,
        'predict_method': 'predict_proba',
        'has_hyperparameter_tuning': True
    },
    'knn': {
        'name': 'KNN',
        'file': 'knn_model.pkl',
        'display_name': 'KNN',
        'use_scaled_data': True,
        'predict_method': 'predict_proba',
        'has_hyperparameter_tuning': True
    },
    'ann': {
        'name': 'ANN',
        'file': 'ann_model.pkl',
        'display_name': 'ANN',
        'use_scaled_data': True,
        'predict_method': 'predict_proba',
        'has_hyperparameter_tuning': True
    }
}

# ================================================================
# 使用示例：
#   - 如果只想训练5个模型（如Logistic、RF、XGB、LGB、ANN）：
#     MODEL_SELECTION = {
#         'logistic': True,
#         'tree': False,
#         'rf': True,
#         'xgb': True,
#         'lgb': True,
#         'svm': False,
#         'adaboost': False,
#         'knn': False,
#         'ann': True
#     }
# ================================================================
# 根据 MODEL_SELECTION 配置选择性训练模型
################# 11.1Logistic模型 ###################
if MODEL_SELECTION['logistic']:
    print("\n" + "="*60)
    print("🚀 开始训练 Logistic 回归模型")
    print("="*60)
    logist_model = sm.Logit(y_train_logist, X_train_logist_const).fit(disp=0)
    logist_model.summary()
    y_train_pred_prob_logist = logist_model.predict(X_train_logist_const)
    auc_logist = roc_auc_score(y_train_logist, y_train_pred_prob_logist)
    print(f"训练集 AUC: {auc_logist:.4f}")
    with open("训练好的模型/logistic_model.pkl", 'wb') as f:
        pickle.dump(logist_model, f)
    print("✓ Logistic 模型训练完成并已保存")
else:
    print("\n⚠ Logistic 模型已禁用，跳过训练")

################ 11.2决策树模型 ############
if MODEL_SELECTION['tree']:
    print("\n" + "="*60)
    print("🚀 开始训练决策树模型")
    print("="*60)
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [5, 10, 20],
        'max_features': ['sqrt', None],
        'ccp_alpha': [0.0, 0.01, 0.1]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    final_model = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
    final_model.fit(X_train, y_train)
    print("最终模型训练完成！")
    
    plt.figure(figsize=(20, 12))
    plot_tree(
        final_model,
        feature_names=X_train.columns.tolist(),
        class_names=['No hosp_expire', 'hosp_expire'],
        filled=True,
        rounded=True,
        fontsize=8,
        proportion=True,
        impurity=False
    )
    plt.title(f'决策树结构 (max_depth={final_model.get_depth()})', fontsize=16)
    plt.tight_layout()
    plt.savefig("决策树图片/tree_structure.jpg", dpi=500, bbox_inches='tight')
    plt.show()
    
    with open("训练好的模型/tree_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ 决策树模型训练完成并已保存")
else:
    print("\n⚠ 决策树模型已禁用，跳过训练")

############ 11.3随机森林(RF)模型 ##########
if MODEL_SELECTION['rf']:
    print("\n" + "="*60)
    print("🚀 开始训练随机森林模型")
    print("="*60)
    param_grid = {
        'n_estimators': np.arange(50, 500, 50),
        'max_features': list(range(2, round(np.sqrt(X_train.shape[1])) + 1))
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    final_model = RandomForestClassifier(**grid_search.best_params_, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)
    print("最终模型训练完成！")
    
    with open("训练好的模型/rf_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ 随机森林模型训练完成并已保存")
else:
    print("\n⚠ 随机森林模型已禁用，跳过训练")

########### 11.4Xgboost模型 ##########
if MODEL_SELECTION['xgb']:
    print("\n" + "="*60)
    print("🚀 开始训练 XGBoost 模型")
    print("="*60)
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = XGBClassifier(
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        nthread=-1,
        use_label_encoder=False
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    final_model = XGBClassifier(
        **grid_search.best_params_,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        nthread=-1,
        use_label_encoder=False
    )
    final_model.fit(X_train, y_train)
    print("最终模型训练完成！")
    
    with open("训练好的模型/xgb_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ XGBoost 模型训练完成并已保存")
else:
    print("\n⚠ XGBoost 模型已禁用，跳过训练")

############# 11.5LightGBM模型 ###########
if MODEL_SELECTION['lgb']:
    print("\n" + "="*60)
    print("🚀 开始训练 LightGBM 模型")
    print("="*60)
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = lgb.LGBMClassifier(random_state=42, objective='binary', n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    final_model = lgb.LGBMClassifier(
        **grid_search.best_params_,
        random_state=42,
        objective='binary',
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    print("最终模型训练完成！")
    
    with open("训练好的模型/lgb_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ LightGBM 模型训练完成并已保存")
else:
    print("\n⚠ LightGBM 模型已禁用，跳过训练")

############ 11.6支持向量机（SVM）模型 #############
if MODEL_SELECTION['svm']:
    print("\n" + "="*60)
    print("🚀 开始训练 SVM 模型")
    print("="*60)
    param_grid = {
        'C': [0.1, 2, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'degree': [2, 3, 4]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = SVC(probability=True)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    final_model = SVC(**grid_search.best_params_, probability=True)
    final_model.fit(X_train, y_train)
    print("最终模型训练完成！")
    
    with open("训练好的模型/svm_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ SVM 模型训练完成并已保存")
else:
    print("\n⚠ SVM 模型已禁用，跳过训练")

############ 11.7AdaBoost模型 #############
if MODEL_SELECTION['adaboost']:
    print("\n" + "="*60)
    print("🚀 开始训练 AdaBoost 模型")
    print("="*60)
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = AdaBoostClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    final_model = AdaBoostClassifier(**grid_search.best_params_, random_state=42)
    final_model.fit(X_train, y_train)
    print("最终模型训练完成！")
    
    with open("训练好的模型/adaboost_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ AdaBoost 模型训练完成并已保存")
else:
    print("\n⚠ AdaBoost 模型已禁用，跳过训练")

############# 11.8K近邻（KNN）模型 ############
if MODEL_SELECTION['knn']:
    print("\n" + "="*60)
    print("🚀 开始训练 KNN 模型")
    print("="*60)
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    final_model = KNeighborsClassifier(**grid_search.best_params_)
    final_model.fit(X_train, y_train)
    print("最终模型训练完成！")
    
    with open("训练好的模型/knn_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ KNN 模型训练完成并已保存")
else:
    print("\n⚠ KNN 模型已禁用，跳过训练")

######### 11.9人工神经网络（ANN）模型 ##########
if MODEL_SELECTION['ann']:
    print("\n" + "="*60)
    print("🚀 开始训练 ANN 模型")
    print("="*60)
    param_grid = {
        'hidden_layer_sizes': [(25,), (50,), (100,), (10, 10), (50, 50), (100,100), (50, 50, 50)],
        'activation': ['relu', 'tanh', 'logistic']
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = MLPClassifier(random_state=123, max_iter=500)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    final_model = MLPClassifier(**grid_search.best_params_, random_state=123, max_iter=500)
    final_model.fit(X_train, y_train)
    print("最终模型训练完成！")
    
    with open("训练好的模型/ann_model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ ANN 模型训练完成并已保存")
else:
    print("\n⚠ ANN 模型已禁用，跳过训练")

# 打印模型训练总结
print("\n" + "="*60)
print("📊 模型训练总结")
print("="*60)
print(f"选中的模型: {SELECTED_MODELS}")
print(f"训练完成的模型数量: {len(SELECTED_MODELS)}")

############12.训练集模型性能##########
########## 12.1加载训练好的模型 ##########
# 根据 SELECTED_MODELS 动态加载模型
models = {}
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    file_path = f"训练好的模型/{config['file']}"
    with open(file_path, 'rb') as f:
        models[model_key] = pickle.load(f)
    print(f"已加载: {config['display_name']}")
print(f"\n总共加载了 {len(models)} 个模型")

######### 12.2.计算训练集预测结果 ###########
# 动态预测所有选中的模型
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    model = models[model_key]
    
    if config['predict_method'] == 'predict':
        y_pred_prob = model.predict(X_train_logist_const if not config['use_scaled_data'] else X_train)
    else:
        y_pred_prob = model.predict_proba(X_train)[:, 1]
    
    y_pred = (y_pred_prob >= 0.5).astype(int)
    globals()[f'y_train_pred_prob_{model_key}'] = y_pred_prob
    globals()[f'y_train_pred_{model_key}'] = y_pred
    print(f"✓ {config['display_name']} 预测完成")

print("\n所有模型预测完成")

########## 12.3计算混淆矩阵并可视化 ############
def CM_plot(cm, title="", save_path=None, show=True):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Survival', 'Death'], 
                yticklabels=['Survival', 'Death'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵图片已保存至: {save_path}")
    if show:
        plt.show()
    plt.close()

cm_save_path = "单模型预测性能/"

# 计算所有选中模型的混淆矩阵
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    cm = confusion_matrix(y_train if config['use_scaled_data'] else y_train_logist, 
                         globals()[f'y_train_pred_{model_key}'])
    globals()[f'cm_{model_key}'] = cm
    CM_plot(cm, title=config['display_name'], save_path=f"{cm_save_path}CM_{config['name']}_训练集.png")

###########12.4计算准确率、精确率、灵敏度、f1分数、特异度 ###########
def calculate_acc_pre_sen_f1_spc(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = tn / (tn + fp)
    return accuracy, precision, sensitivity, f1score, specificity

results_list = []
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    cm = globals()[f'cm_{model_key}']
    accuracy, precision, sensitivity, f1score, specificity = calculate_acc_pre_sen_f1_spc(cm)
    print(f"{config['display_name']} → Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Sensitivity: {sensitivity:.3f}, F1: {f1score:.3f}, Specificity: {specificity:.3f}")
    results_list.append({
        'Model': config['display_name'],
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'F1 Score': f1score,
        'Specificity': specificity
    })
    globals()[f'accuracy_{model_key}'] = accuracy
    globals()[f'precision_{model_key}'] = precision
    globals()[f'sensitivity_{model_key}'] = sensitivity
    globals()[f'f1_{model_key}'] = f1score
    globals()[f'specificity_{model_key}'] = specificity

results_df = pd.DataFrame(results_list)
print(results_df)
results_df.to_csv("所有模型预测性能/model_performance_metrics.csv", index=False, encoding='utf-8-sig')

############# 12.5计算AUC及其95%置信区间 ########
def calculate_auc(y_label, y_pred_prob):
    auc_value = roc_auc_score(y_label, y_pred_prob)
    se_auc = np.sqrt((auc_value * (1 - auc_value)) / len(y_label))
    z = norm.ppf(0.975)
    auc_ci_lower = auc_value - z * se_auc
    auc_ci_upper = auc_value + z * se_auc
    return auc_value, auc_ci_lower, auc_ci_upper

for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_train if config['use_scaled_data'] else y_train_logist
    y_pred = globals()[f'y_train_pred_prob_{model_key}']
    auc_value, auc_ci_lower, auc_ci_upper = calculate_auc(y_true, y_pred)
    globals()[f'auc_value_{model_key}'] = auc_value
    globals()[f'auc_ci_lower_{model_key}'] = auc_ci_lower
    globals()[f'auc_ci_upper_{model_key}'] = auc_ci_upper
    print(f"{config['display_name']} AUC: {auc_value:.3f} (95% CI: {auc_ci_lower:.3f} - {auc_ci_upper:.3f})")

#############12.6绘制ROC曲线 ###########
def ROC_plot(y_label, y_pred_prob, auc_value, auc_ci_lower, auc_ci_upper, name="", save_path="单模型预测性能/"):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{name}: AUC = {auc_value:.3f} (95% CI: {auc_ci_lower:.3f} - {auc_ci_upper:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - " + name)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}ROC_{name}_训练集.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ {name} 模型ROC曲线已保存")

# 绘制所有选中模型的ROC曲线
models_roc = []
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_train if config['use_scaled_data'] else y_train_logist
    models_roc.append((
        config['name'],
        y_true,
        globals()[f'y_train_pred_prob_{model_key}'],
        globals()[f'auc_value_{model_key}'],
        globals()[f'auc_ci_lower_{model_key}'],
        globals()[f'auc_ci_upper_{model_key}']
    ))

for name, y_label, y_pred_prob, auc_val, auc_ci_l, auc_ci_u in models_roc:
    ROC_plot(y_label, y_pred_prob, auc_val, auc_ci_l, auc_ci_u, name)

print("\n✓ 所有模型的训练集ROC曲线已保存完成")

# 绘制所有模型在同一张图
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '--']

for idx, (name, y_label, y_pred_prob, auc_value, auc_ci_l, auc_ci_u) in enumerate(models_roc):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.plot(fpr, tpr, 
             label=f'{name}: AUC = {auc_value:.3f} (95% CI: {auc_ci_l:.3f}-{auc_ci_u:.3f})',
             color=colors[idx % len(colors)], 
             linestyle=linestyles[idx % len(linestyles)],
             linewidth=2)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Chance', linewidth=1.5)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves for All Models (Training Set)", fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("所有模型预测性能/ROC_curves_allmodel_training.png", dpi=300)
plt.show()
print("所有模型的ROC曲线已保存至: 所有模型预测性能/ROC_curves_allmodel_training.png")

############# 12.7绘制校准曲线 ###########
# 单个模型校准曲线
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_train if config['use_scaled_data'] else y_train_logist
    y_pred_prob = globals()[f'y_train_pred_prob_{model_key}']
    plt.figure(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y_true, y_pred_prob, n_bins=10, marker='o', label=config['name'])
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve - {config['name']}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"单模型预测性能/Calibration_{config['name']}_训练集.png", dpi=300)
    plt.show()
    print(f"✓ {config['display_name']} 校准曲线已保存")

# 所有模型在同一张图
plt.figure(figsize=(12, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

for idx, model_key in enumerate(SELECTED_MODELS):
    config = MODEL_CONFIG[model_key]
    y_true = y_train if config['use_scaled_data'] else y_train_logist
    y_pred_prob = globals()[f'y_train_pred_prob_{model_key}']
    CalibrationDisplay.from_predictions(
        y_true, y_pred_prob, n_bins=10, marker=markers[idx], 
        label=config['name'], color=colors[idx], ax=plt.gca()
    )

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curves for All Models (Training Set)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("所有模型预测性能/Calibration_curves_allmodel_training.png", dpi=300)
plt.show()
print("✓ 所有模型校准曲线对比图已保存")

# Brier Score
print("=== Brier Score (训练集) ===")
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_train if config['use_scaled_data'] else y_train_logist
    y_pred_prob = globals()[f'y_train_pred_prob_{model_key}']
    bs = brier_score_loss(y_true, y_pred_prob)
    print(f"{config['display_name']}: {bs:.4f}")

############# 12.8绘制决策分析曲线 (DCA) #############
def calculate_net_benefit(y_label, y_pred_prob, thresholds=np.linspace(0.01, 0.99, 100)):
    net_benefit_model = []
    net_benefit_alltrt = []
    net_benefits_notrt = [0] * len(thresholds)
    total_obs = len(y_label)
    for thresh in thresholds:
        y_pred_label = (y_pred_prob > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        net_benefit = (tp / total_obs) - (fp / total_obs) * (thresh / (1 - thresh))
        net_benefit_model.append(net_benefit)
        tn_all, fp_all, fn_all, tp_all = confusion_matrix(y_label, y_label).ravel()
        total_right = tp_all + tn_all
        net_benefit_all = (tp_all / total_right) - (tn_all / total_right) * (thresh / (1 - thresh))
        net_benefit_alltrt.append(net_benefit_all)
    return net_benefit_model, net_benefit_alltrt, net_benefits_notrt

def plot_DCA(model_data_list=None, thresholds=np.linspace(0.01, 0.99, 100), 
             save_path=None, show=True, title="Decision Curve Analysis"):
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    all_net_benefits = []
    
    if model_data_list is not None:
        for idx, (m_name, y_true_m, y_pred_prob_m) in enumerate(model_data_list):
            net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefit(y_true_m, y_pred_prob_m, thresholds)
            all_net_benefits.extend(net_benefit)
            plt.plot(thresholds, net_benefit, label=m_name, color=colors[idx % len(colors)], linewidth=2)
    
    plt.plot(thresholds, net_benefit_alltrt, label="Treat All", color="red", linewidth=2, linestyle='--')
    plt.plot(thresholds, net_benefits_notrt, label="Treat None", color="green", linewidth=2, linestyle='--')
    
    if all_net_benefits:
        plt.ylim(min(-0.15, min(all_net_benefits) - 0.05), max(0.3, max(all_net_benefits) + 0.1))
    
    plt.xlabel("Threshold Probability", fontsize=12)
    plt.ylabel("Net Benefit", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

# DCA多模型对比图
dca_model_list = []
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_train if config['use_scaled_data'] else y_train_logist
    y_pred_prob = globals()[f'y_train_pred_prob_{model_key}']
    dca_model_list.append((config['name'], y_true, y_pred_prob))

plot_DCA(
    model_data_list=dca_model_list,
    save_path="所有模型预测性能/DCA_curves_allmodel_training.png",
    title="Decision Curve Analysis for All Models (Training Set)"
)

# 单个模型DCA
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_train if config['use_scaled_data'] else y_train_logist
    y_pred_prob = globals()[f'y_train_pred_prob_{model_key}']
    plot_DCA(
        model_data_list=[(config['name'], y_true, y_pred_prob)],
        save_path=f"单模型预测性能/DCA_{config['name']}_training.png",
        title=f"Decision Curve Analysis - {config['name']} (Training Set)"
    )
    print(f"✓ {config['display_name']} DCA曲线已保存")

#############13.测试集数据#######
######13.1加载训练好的模型 ####
models_test = {}
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    file_path = f"训练好的模型/{config['file']}"
    with open(file_path, 'rb') as f:
        models_test[model_key] = pickle.load(f)
print("✓ 所有模型加载完成")

############ 13.2计算测试数据集预测结果 ##########
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    model = models_test[model_key]
    
    if config['predict_method'] == 'predict':
        y_pred_prob = model.predict(X_test_logist_const if not config['use_scaled_data'] else X_test)
    else:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_pred_prob >= 0.5).astype(int)
    globals()[f'y_test_pred_prob_{model_key}'] = y_pred_prob
    globals()[f'y_test_pred_{model_key}'] = y_pred
    print(f"✓ {config['display_name']} 测试集预测完成")

########## 13.3计算测试集混淆矩阵 ############
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_test if config['use_scaled_data'] else y_test_logist
    cm = confusion_matrix(y_true, globals()[f'y_test_pred_{model_key}'])
    globals()[f'cm_{model_key}_test'] = cm
    CM_plot(cm, title=f"{config['display_name']}", save_path=f"{cm_save_path}CM_{config['name']}_测试集.png")

###########13.4计算测试集准确率、精确率、灵敏度、f1分数、特异度 ##########
results_list_test = []
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    cm = globals()[f'cm_{model_key}_test']
    accuracy, precision, sensitivity, f1score, specificity = calculate_acc_pre_sen_f1_spc(cm)
    print(f"{config['display_name']} → Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Sensitivity: {sensitivity:.3f}, F1: {f1score:.3f}, Specificity: {specificity:.3f}")
    results_list_test.append({
        'Model': config['display_name'],
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'F1 Score': f1score,
        'Specificity': specificity
    })

results_df_test = pd.DataFrame(results_list_test)
print(results_df_test)
results_df_test.to_csv("所有模型预测性能/model_performance_metrics_test.csv", index=False, encoding='utf-8-sig')

############# 13.5计算测试集AUC及其95%置信区间 ########
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_test if config['use_scaled_data'] else y_test_logist
    y_pred = globals()[f'y_test_pred_prob_{model_key}']
    auc_value, auc_ci_lower, auc_ci_upper = calculate_auc(y_true, y_pred)
    globals()[f'auc_value_{model_key}_test'] = auc_value
    globals()[f'auc_ci_lower_{model_key}_test'] = auc_ci_lower
    globals()[f'auc_ci_upper_{model_key}_test'] = auc_ci_upper
    print(f"{config['display_name']} AUC: {auc_value:.3f} (95% CI: {auc_ci_lower:.3f} - {auc_ci_upper:.3f})")

########### 13.6.绘制ROC曲线（测试集）############
# 单个模型ROC
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_test if config['use_scaled_data'] else y_test_logist
    y_pred_prob = globals()[f'y_test_pred_prob_{model_key}']
    auc_val = globals()[f'auc_value_{model_key}_test']
    auc_ci_l = globals()[f'auc_ci_lower_{model_key}_test']
    auc_ci_u = globals()[f'auc_ci_upper_{model_key}_test']
    ROC_plot(y_true, y_pred_prob, auc_val, auc_ci_l, auc_ci_u, f"{config['name']} (Test Set)")

# 所有模型在同一张图
models_roc_test = []
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_test if config['use_scaled_data'] else y_test_logist
    models_roc_test.append((
        config['name'],
        y_true,
        globals()[f'y_test_pred_prob_{model_key}'],
        globals()[f'auc_value_{model_key}_test'],
        globals()[f'auc_ci_lower_{model_key}_test'],
        globals()[f'auc_ci_upper_{model_key}_test']
    ))

plt.figure(figsize=(12, 9))
for idx, (name, y_label, y_pred_prob, auc_value, auc_ci_l, auc_ci_u) in enumerate(models_roc_test):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.plot(fpr, tpr, 
             label=f'{name}: AUC = {auc_value:.3f} (95% CI: {auc_ci_l:.3f}-{auc_ci_u:.3f})',
             color=colors[idx % len(colors)], 
             linestyle=linestyles[idx % len(linestyles)],
             linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Chance', linewidth=1.5)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves for All Models (Test Set)", fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("所有模型预测性能/ROC_curves_allmodel_test.png", dpi=300)
plt.show()
print("✓ 测试集ROC曲线对比图已保存")

########## 13.7 绘制测试集校准曲线 ########
# 单个模型校准曲线
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_test if config['use_scaled_data'] else y_test_logist
    y_pred_prob = globals()[f'y_test_pred_prob_{model_key}']
    plt.figure(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y_true, y_pred_prob, n_bins=10, marker='o', label=config['name'])
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve - {config['name']} (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"单模型预测性能/Calibration_{config['name']}_test.png", dpi=300)
    plt.show()

# 所有模型在同一张图
plt.figure(figsize=(12, 10))
for idx, model_key in enumerate(SELECTED_MODELS):
    config = MODEL_CONFIG[model_key]
    y_true = y_test if config['use_scaled_data'] else y_test_logist
    y_pred_prob = globals()[f'y_test_pred_prob_{model_key}']
    CalibrationDisplay.from_predictions(
        y_true, y_pred_prob, n_bins=10, marker=markers[idx], 
        label=config['name'], color=colors[idx], ax=plt.gca()
    )
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curves for All Models (Test Set)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("所有模型预测性能/Calibration_curves_allmodel_test.png", dpi=300)
plt.show()
print("✓ 测试集校准曲线对比图已保存")

######### 13.8绘制测试集DCA曲线 ########
# 多模型对比
dca_model_list_test = []
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_test if config['use_scaled_data'] else y_test_logist
    y_pred_prob = globals()[f'y_test_pred_prob_{model_key}']
    dca_model_list_test.append((config['name'], y_true, y_pred_prob))

plot_DCA(
    model_data_list=dca_model_list_test,
    save_path="所有模型预测性能/DCA_curves_allmodel_test.png",
    title="Decision Curve Analysis for All Models (Test Set)"
)

# 单个模型DCA
for model_key in SELECTED_MODELS:
    config = MODEL_CONFIG[model_key]
    y_true = y_test if config['use_scaled_data'] else y_test_logist
    y_pred_prob = globals()[f'y_test_pred_prob_{model_key}']
    plot_DCA(
        model_data_list=[(config['name'], y_true, y_pred_prob)],
        save_path=f"单模型预测性能/DCA_{config['name']}_test.png",
        title=f"Decision Curve Analysis - {config['name']} (Test Set)"
    )
    print(f"✓ {config['display_name']} 测试集DCA曲线已保存")

print("\n" + "="*60)
print("✅ 所有模型评估完成！")
print("="*60)

############# 14. 模型可解释性 (SHAP) #############
save_path = "SHAP结果文件/" # 图片保存路径
## 编写绘制可解释性图片的函数，方便调用 ##
def shap_explain_model(model, X_train, X_test, model_name="logist", n_interpret=50, obs_index=5, n_top_features=6):
    """
使用 SHAP 进行模型的可解释性分析，并保存相关图像。
参数:
model: 训练好的机器学习模型
X_train: 训练数据
X_test: 测试数据（包含常数项）
model_name: 模型名称（用于保存文件名）
n_interpret: 用于 SHAP 解释的数据样本量（按照自己需求更改,不能超过测试集的样本量）
obs_index: 选择哪一条观测用于瀑布图和单个观测的力图 （按照自己需求更改,不能超过n_interpret)
n_top_features: Dependence Plot显示的重要特征数量默认值为6,可设置为4, 6, 8, 10等
"""
    if model_name == "logist":
        # 1. 设定模型预测函数
        def model_predict(X):
            return model.predict(X)
        # 2. 建立 SHAP 核解释器（KernelExplainer 适用于任何模型）
        explainer = shap.KernelExplainer(model_predict, X_train)
        shap_values = explainer(X_test.iloc[:n_interpret, :])
        # 3. 生成 SHAP 解释性图像并保存 #
        os.makedirs(save_path, exist_ok=True)
        # 3.1 特征重要性条形图（全局 SHAP 重要性）
        plt.figure(figsize=(15, 12))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.savefig(f"{save_path}bar_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 3.2 特征重要性蜂群图
        plt.figure(figsize=(15, 12))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.savefig(f"{save_path}beeswarm_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 3.3 瀑布图（展示选定观测的数据解释）
        plt.figure(figsize=(15, 12))
        shap.plots.waterfall(shap_values[obs_index], max_display=15, show=False)
        plt.savefig(f"{save_path}waterfall_{obs_index}_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 3.4 力图（选定观测）
        force_plot_sig = shap.plots.force(shap_values[obs_index, :])
        shap.save_html(f"{save_path}force_plot_sig_{obs_index}_{model_name}.html", force_plot_sig)
        # 3.5 力图（所有观测）
        force_plot_all = shap.plots.force(shap_values)
        shap.save_html(f"{save_path}force_plot_all_{model_name}.html", force_plot_all)
        
        # 3.6 SHAP Dependence Plot（依赖图）- Logistic模型
        print("\n📊 3.7 生成SHAP Dependence Plots...")
        # 获取特征名称列表
        feature_names = list(shap_values.feature_names)
        # 选择最重要的n_top_features个特征进行依赖图分析
        shap_mean = np.abs(shap_values.values).mean(0)
        n_features = min(n_top_features, len(feature_names))  # 确保不超过特征总数
        top_features_indices = np.argsort(-shap_mean)[:n_features]
        top_features = [feature_names[i] for i in top_features_indices]
        
        print(f"选择的{n_features}个重要特征: {top_features}")
        
        # 根据n_features计算子图网格的行列数
        n_cols = int(np.ceil(np.sqrt(n_features)))  # 列数：向上取整平方根
        n_rows = int(np.ceil(n_features / n_cols))  # 行数：向上取整
        fig_size = (5 * n_cols, 4 * n_rows)  # 根据网格大小动态调整图片尺寸
        
        # 创建子图网格
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        if n_features == 1:
            axes = np.array([axes])  # 确保axes是数组
        axes = axes.flatten()
        
        # 获取用于绑定的特征索引
        X_test_sample = X_test.iloc[:n_interpret, :].copy()
        
        for i, feature_idx in enumerate(top_features_indices):
            # 尝试找到最佳绑定特征
            if len(top_features_indices) > 1 and i < len(top_features_indices):
                other_indices = [idx for j, idx in enumerate(top_features_indices) if j != i]
                if len(other_indices) > 0:
                    binding_feature_idx = other_indices[0]
                else:
                    binding_feature_idx = None
            else:
                binding_feature_idx = None
            
            # 绘制依赖图
            plt.sca(axes[i])
            shap.dependence_plot(
                feature_idx,
                shap_values.values,
                X_test_sample,
                feature_names=feature_names,
                interaction_index=binding_feature_idx,
                ax=axes[i],
                show=False
            )
            axes[i].set_title(f'SHAP Dependence: {feature_names[feature_idx]}', 
                            fontweight='bold', fontsize=12)
            # 清除colorbar上的变量名称文本，仅保留颜色渐变条
            for coll in axes[i].collections:
                if hasattr(coll, 'colorbar') and coll.colorbar is not None:
                    coll.colorbar.set_label('')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}dependence_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SHAP Dependence Plot 已保存至: {save_path}dependence_{model_name}.png")

    if model_name != "logist":
        # 1. 建立 SHAP 核解释器（KernelExplainer 适用于任何模型）
        explainer = shap.KernelExplainer(lambda X: model.predict_proba(X), X_train)
        shap_values = explainer(X_test.iloc[:n_interpret, :])
        # 2. 生成 SHAP 解释性图像并保存
        os.makedirs(save_path, exist_ok=True)
        # 2.1 特征重要性条形图（全局 SHAP 重要性）
        plt.figure(figsize=(15, 12))
        shap.plots.bar(shap_values[:,:,1], max_display=15, show=False)
        plt.savefig(f"{save_path}bar_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 2.2 特征重要性蜂群图
        plt.figure(figsize=(15, 12))
        shap.plots.beeswarm(shap_values[:,:,1], max_display=15, show=False)
        plt.savefig(f"{save_path}beeswarm_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 2.3 瀑布图（展示选定观测的数据解释）
        plt.figure(figsize=(15, 12))
        shap.plots.waterfall(shap_values[obs_index][:,1], max_display=15, show=False)
        plt.savefig(f"{save_path}waterfall_{obs_index}_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 2.4 力图（选定观测）
        force_plot_sig = shap.plots.force(shap_values[obs_index, :, 1])
        shap.save_html(f"{save_path}force_plot_sig_{obs_index}_{model_name}.html", force_plot_sig)
        # 2.5 力图（所有观测）
        force_plot_all = shap.plots.force(shap_values[:, :, 1])
        shap.save_html(f"{save_path}force_plot_all_{model_name}.html", force_plot_all)
        
        # 2.6 SHAP Dependence Plot（依赖图）- 非Logistic模型
        print("\n📊 2.7 生成SHAP Dependence Plots...")
        # 获取特征名称列表
        feature_names = list(shap_values.feature_names)
        # 选择最重要的n_top_features个特征进行依赖图分析
        shap_mean = np.abs(shap_values.values[:, :, 1]).mean(0)
        n_features = min(n_top_features, len(feature_names))  # 确保不超过特征总数
        top_features_indices = np.argsort(-shap_mean)[:n_features]
        top_features = [feature_names[i] for i in top_features_indices]
        
        print(f"选择的{n_features}个重要特征: {top_features}")
        
        # 根据n_features计算子图网格的行列数
        n_cols = int(np.ceil(np.sqrt(n_features)))  # 列数：向上取整平方根
        n_rows = int(np.ceil(n_features / n_cols))  # 行数：向上取整
        fig_size = (5 * n_cols, 4 * n_rows)  # 根据网格大小动态调整图片尺寸
        
        # 创建子图网格
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        if n_features == 1:
            axes = np.array([axes])  # 确保axes是数组
        axes = axes.flatten()
        
        # 获取用于绑定的特征数据
        X_test_sample = X_test.iloc[:n_interpret, :].copy()
        
        for i, feature_idx in enumerate(top_features_indices):
            # 尝试找到最佳绑定特征
            if len(top_features_indices) > 1 and i < len(top_features_indices):
                other_indices = [idx for j, idx in enumerate(top_features_indices) if j != i]
                if len(other_indices) > 0:
                    binding_feature_idx = other_indices[0]
                else:
                    binding_feature_idx = 'auto'
            else:
                binding_feature_idx = 'auto'
            
            # 绘制依赖图
            plt.sca(axes[i])
            shap.dependence_plot(
                feature_idx,
                shap_values.values[:, :, 1],
                X_test_sample,
                feature_names=feature_names,
                interaction_index=binding_feature_idx,
                ax=axes[i],
                show=False
            )
            axes[i].set_title(f'SHAP Dependence: {feature_names[feature_idx]}', 
                            fontweight='bold', fontsize=12)
            # 清除colorbar上的变量名称文本，仅保留颜色渐变条
            for coll in axes[i].collections:
                if hasattr(coll, 'colorbar') and coll.colorbar is not None:
                    coll.colorbar.set_label('')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}dependence_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SHAP Dependence Plot 已保存至: {save_path}dependence_{model_name}.png")
    
    print(f"所有 SHAP 解释性图像已保存在 {save_path}")

# 14.1 Logistic模型的可解释性 
shap_explain_model(model=models['logistic'], X_train=X_train_logist_const, X_test=X_test_logist_const, 
                   model_name="logist", n_interpret=200, obs_index=8, n_top_features=6)

# 14.2. 决策树模型的可解释性 
shap_explain_model(model=models['tree'], X_train=X_train, X_test=X_test, 
                   model_name="tree", n_interpret=250, obs_index=8, n_top_features=8)

# 14.3. 随机森林模型的可解释性 
shap_explain_model(model=models['rf'], X_train=X_train, X_test=X_test, 
                   model_name="rf", n_interpret=50, obs_index=5, n_top_features=6)

# 14.4. XGBoost模型的可解释性 
shap_explain_model(model=models['xgb'], X_train=X_train, X_test=X_test, 
                   model_name="xgb", n_interpret=100, obs_index=13, n_top_features=6)

# 14.5. LightGBM模型的可解释性 
shap_explain_model(model=models['lgb'], X_train=X_train, X_test=X_test, 
                   model_name="lgb", n_interpret=50, obs_index=5, n_top_features=6)

# 14.6. SVM模型的可解释性 
shap_explain_model(model=models['svm'], X_train=X_train, X_test=X_test, 
                   model_name="svm", n_interpret=50, obs_index=5, n_top_features=6)

# 14.7. ANN模型的可解释性 
shap_explain_model(model=models['ann'], X_train=X_train, X_test=X_test, 
                   model_name="ann", n_interpret=300, obs_index=5, n_top_features=4)

# 14.8. AdaBoost模型的可解释性 
shap_explain_model(model=models['adaboost'], X_train=X_train, X_test=X_test, 
                   model_name="adaboost", n_interpret=50, obs_index=5, n_top_features=6)

# 14.9. KNN模型的可解释性 
shap_explain_model(model=models['knn'], X_train=X_train, X_test=X_test, 
                   model_name="knn", n_interpret=50, obs_index=5, n_top_features=6)

print("\n" + "="*60)
print("✅ 所有模型SHAP可解释性分析完成!")
print("="*60)


############# 15. SHAP交互作用热力图（独立函数） #############
# 定义SHAP交互作用热力图函数，支持九大模型任意一模型
def shap_interaction_heatmap(model, X_train, X_test, model_name="model", n_samples=50, class_idx=1):
    """
    生成SHAP交互作用热力图,支持任意机器学习模型
    
    参数:
        model: 训练好的机器学习模型
        X_train: 训练数据用于创建SHAP解释器
        X_test: 测试数据用于计算SHAP值
        model_name: 模型名称（用于保存文件名），默认为"model"
        n_samples: 用于计算SHAP交互值的样本数量默认50,不宜过大
        class_idx: 类别索引,对于二分类模型通常取1正类,默认为1
    """
    import warnings
    warnings.filterwarnings("ignore")
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name} 模型 SHAP交互作用热力图分析")
    print(f"{'='*60}")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 确保样本数量不超过测试集大小
    n_samples = min(n_samples, len(X_test))
    print(f"📌 使用样本数量: {n_samples}")
    
    # 限制样本数量以避免计算时间过长
    if n_samples > 100:
        print("⚠ 样本量较大，可能需要较长的计算时间...")
        print("  建议: n_samples <= 50 可获得较好的计算效率")
    
    X_sample = X_test.iloc[:n_samples].copy()
    
    # 判断是否为Logistic模型（使用predict方法）
    is_logistic = model_name.lower() in ['logist', 'logistic']
    
    # 判断是否为树模型（可以使用TreeExplainer）
    tree_models = ['tree', 'decision_tree', 'rf', 'random_forest', 'xgb', 'xgboost', 
                   'lgb', 'lightgbm', 'adaboost', 'gbm', 'gradient_boosting']
    is_tree_model = any(model_name.lower() == tm or model_name.lower().replace('_', '').replace(' ', '') == tm.replace('_', '')
                        for tm in tree_models)
    
    try:
        # 1. 创建SHAP解释器
        print("\n🔄 步骤1: 创建SHAP解释器...")
        
        if is_tree_model:
            # 树模型使用TreeExplainer（支持shap_interaction_values）
            print("  使用TreeExplainer（适用于树模型）")
            explainer = shap.TreeExplainer(model)
            print("✓ SHAP解释器创建成功")
            
            # 2. 计算SHAP交互值
            print(f"\n🔄 步骤2: 计算SHAP交互值 (样本数: {n_samples})...")
            print("  这可能需要一些时间，请耐心等待...")
            
            shap_interaction_values = explainer.shap_interaction_values(X_sample)
            
            # 处理二分类模型的输出（取正类）
            # shap_interaction_values 可能是 (n_samples, n_features, n_features, n_classes) 或 list
            if isinstance(shap_interaction_values, list):
                # 如果是列表，取正类
                shap_interaction_values = shap_interaction_values[class_idx]
                print("  - 已提取正类(类别1)的交互值")
            elif len(shap_interaction_values.shape) == 4:
                # 如果是4D数组 (n_samples, n_features, n_features, n_classes)，取正类
                shap_interaction_values = shap_interaction_values[:, :, :, class_idx]
                print("  - 已提取正类(类别1)的交互值")
            
            print(f"✓ SHAP交互值计算完成")
            print(f"  - 交互值矩阵形状: {shap_interaction_values.shape}")
            
            # 3. 计算平均交互作用矩阵
            print("\n📊 步骤3: 计算平均交互作用矩阵...")
            interaction_matrix = np.abs(shap_interaction_values).mean(0)
            print(f"  - 交互矩阵形状: {interaction_matrix.shape}")
            
        else:
            # 非树模型使用KernelExplainer + 手动计算近似交互值
            print("  使用KernelExplainer + 近似交互值计算")
            
            if is_logistic:
                # Logistic模型
                def model_predict(X):
                    return model.predict(X)
                explainer = shap.KernelExplainer(model_predict, X_train)
            else:
                # 其他模型使用predict_proba
                explainer = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, class_idx], X_train)
            
            print("✓ SHAP解释器创建成功")
            
            # 2. 计算SHAP值（非交互值）
            print(f"\n🔄 步骤2: 计算SHAP值 (样本数: {n_samples})...")
            print("  这可能需要一些时间，请耐心等待...")
            
            shap_values = explainer.shap_values(X_sample)
            
            # 处理二分类模型的输出（取正类）
            if isinstance(shap_values, list):
                shap_values = shap_values[class_idx]
                print("  - 已提取正类(类别1)的SHAP值")
            
            print(f"✓ SHAP值计算完成")
            print(f"  - SHAP值矩阵形状: {shap_values.shape}")
            
            # 3. 手动计算近似交互作用矩阵
            print("\n📊 步骤3: 计算近似交互作用矩阵...")
            # 方法：使用SHAP值的乘积的绝对值作为交互作用的近似度量
            n_features = shap_values.shape[1]
            interaction_matrix = np.zeros((n_features, n_features))
            
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        # 对角线：使用SHAP值的方差作为主效应
                        interaction_matrix[i, j] = np.abs(shap_values[:, i]).mean()
                    else:
                        # 非对角线：使用SHAP值乘积的绝对值均值
                        interaction_matrix[i, j] = np.abs(shap_values[:, i] * shap_values[:, j]).mean()
            
            print(f"  - 交互矩阵形状: {interaction_matrix.shape}")
        
        # 4. 获取特征名称
        if hasattr(X_sample, 'columns'):
            feature_names = list(X_sample.columns)
        else:
            feature_names = [f"Feature_{i}" for i in range(interaction_matrix.shape[0])]
        
        print(f"  - 特征数量: {len(feature_names)}")
        
        # 5. 绘制交互作用热力图
        print("\n🎨 步骤4: 绘制交互作用热力图...")
        
        # 创建保存目录
        save_path = "SHAP结果文件/" #保存路径
        os.makedirs(save_path, exist_ok=True)
        
        # 创建图形 - 两个子图
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 5.1 完整热力图
        ax1 = axes[0]
        sns.heatmap(
            interaction_matrix,
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax1,
            annot_kws={'size': 8}
        )
        ax1.set_title(f'{model_name} - SHAP Feature Interaction Matrix (Full)', 
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Feature', fontsize=12)
        ax1.set_ylabel('Feature', fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.tick_params(axis='y', rotation=0, labelsize=9)
        
        # 5.2 上三角热力图（避免重复信息）
        ax2 = axes[1]
        
        # 创建上三角掩码
        mask = np.triu(np.ones_like(interaction_matrix, dtype=bool), k=1)
        
        # 使用掩码绘制热力图
        sns.heatmap(
            interaction_matrix,
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax2,
            mask=mask,  # 应用上三角掩码
            annot_kws={'size': 8}
        )
        ax2.set_title(f'{model_name} - SHAP Feature Interaction Matrix (Upper Triangle)', 
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Feature', fontsize=12)
        ax2.set_ylabel('Feature', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.tick_params(axis='y', rotation=0, labelsize=9)
        
        plt.tight_layout()
        
        # 保存图片
        save_file = f"{save_path}shap_interaction_heatmap_{model_name}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ 热力图已保存至: {save_file}")
        
        # 显示图片
        plt.show()
        plt.close()
        
        # 6. 保存交互值矩阵到CSV（可选）
        interaction_df = pd.DataFrame(
            interaction_matrix,
            index=feature_names,
            columns=feature_names
        )
        csv_file = f"{save_path}shap_interaction_matrix_{model_name}.csv"
        interaction_df.to_csv(csv_file, encoding='utf-8-sig')
        print(f"✓ 交互值矩阵已保存至: {csv_file}")
        
        # 7. 打印交互作用最强的特征对
        print("\n📋 特征交互作用排名 (Top 10):")
        
        # 提取上三角部分的交互值（排除对角线）
        upper_tri_indices = np.triu_indices(len(feature_names), k=1)
        interaction_pairs = []
        for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
            interaction_pairs.append({
                'Feature_1': feature_names[i],
                'Feature_2': feature_names[j],
                'Interaction_Value': interaction_matrix[i, j]
            })
        
        # 按交互值排序
        interaction_pairs_sorted = sorted(interaction_pairs, 
                                         key=lambda x: x['Interaction_Value'], 
                                         reverse=True)
        
        # 显示前10对
        for idx, pair in enumerate(interaction_pairs_sorted[:10], 1):
            print(f"  {idx}. {pair['Feature_1']} x {pair['Feature_2']}: {pair['Interaction_Value']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"✓ {model_name} SHAP交互作用热力图生成完成!")
        print(f"{'='*60}")
        
        return interaction_matrix, feature_names
        
    except Exception as e:
        print(f"\n❌ 生成SHAP交互作用热力图时出错: {e}")
        print("\n💡 建议解决方案:")
        print("  1. 减少n_samples参数值（建议 <= 50）")
        print("  2. 检查模型是否正确加载")
        print("  3. 确保X_train和X_test的数据格式正确")
        import traceback
        traceback.print_exc()
        return None, None


# ==================== 调用示例 ====================
# 以下是九大模型的SHAP交互作用热力图生成示例
# 取消注释即可使用

# 15.1 Logistic模型
shap_interaction_heatmap(
    model=models['logistic'],
    X_train=X_train_logist_const,
    X_test=X_test_logist_const,
    model_name="Logistic",
    n_samples=50
)

# 15.2 决策树模型
shap_interaction_heatmap(
    model=models['tree'],
    X_train=X_train,
    X_test=X_test,
    model_name="Decision_Tree",
    n_samples=250
)

# 15.3 随机森林模型
shap_interaction_heatmap(
     model=models['rf'],
     X_train=X_train,
     X_test=X_test,
     model_name="Random_Forest",
     n_samples=30
 )

# 15.4 XGBoost模型
shap_interaction_heatmap(
     model=models['xgb'],
     X_train=X_train,
     X_test=X_test,
     model_name="XGBoost",
     n_samples=100
 )

# 15.5 LightGBM模型
shap_interaction_heatmap(
     model=models['lgb'],
     X_train=X_train,
     X_test=X_test,
     model_name="LightGBM",
     n_samples=30
 )

# 15.6 SVM模型
shap_interaction_heatmap(
     model=models['svm'],
     X_train=X_train,
     X_test=X_test,
     model_name="SVM",
     n_samples=30
 )

# 15.7 AdaBoost模型
shap_interaction_heatmap(
     model=models['adaboost'],
     X_train=X_train,
     X_test=X_test,
     model_name="AdaBoost",
     n_samples=30
 )

# 15.8 KNN模型 
shap_interaction_heatmap(
     model=models['knn'],
     X_train=X_train,
     X_test=X_test,
     model_name="KNN",
     n_samples=30
 )

# 15.9 ANN模型 
shap_interaction_heatmap(
     model=models['ann'],
     X_train=X_train,
     X_test=X_test,
     model_name="ANN",
     n_samples=30
 )
