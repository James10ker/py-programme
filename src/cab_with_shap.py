import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # 新增: 导入SHAP库


def fix_matplotlib_display():
    """解决matplotlib中文显示问题"""
    import matplotlib as mpl

    # 设置全局字体 - 选择支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 设置更大的标题字号
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14

    # 增加图表边距，避免文字被截断
    plt.rcParams['figure.subplot.top'] = 0.85
    plt.rcParams['figure.subplot.bottom'] = 0.15
    plt.rcParams['figure.subplot.left'] = 0.15
    plt.rcParams['figure.subplot.right'] = 0.95

    # 检查matplotlib后端
    print(f"当前Matplotlib后端: {mpl.get_backend()}")


def read_file(path):
    """读取数据文件"""
    df = pd.read_csv(path, sep=',')
    return df


def preprocess_data(df):
    """数据预处理"""
    # 检查原始数据的Churn分布
    print("原始Churn列值计数:")
    print(df['Churn'].value_counts())

    # 检查所有特征
    print(f"原始数据特征列表: {df.columns.tolist()}")
    print(f"原始特征数量: {len(df.columns)}")

    # 删除指定列
    df = df.drop(columns=["Partner", "Dependents", "PaperlessBilling",
                          "PaymentMethod", "customerID", "gender"])

    print(f"处理后特征列表: {df.columns.tolist()}")
    print(f"保留特征数量: {len(df.columns)}")

    # 处理Churn目标变量 - 必须在其他处理前进行
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    if df['Churn'].isna().any():
        print("警告: Churn列存在无法映射的值")
        df['Churn'] = df['Churn'].fillna(0)

    # 处理其他列的Yes/No值
    for col in df.columns:
        if col != 'Churn':  # 避免重复处理Churn
            df.loc[df[col] == "Yes", col] = 1
            df.loc[df[col] == 'No', col] = 0

    # 确保SeniorCitizen为数值型
    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')

    # 确保tenure为数值型
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

    # 处理其他分类变量
    categorical_features = ['PhoneService', 'MultipleLines', 'InternetService',
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']

    # 将分类变量转换为数值型
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # 处理TotalCharges和MonthlyCharges中的空值或非数值数据
    for col in ['TotalCharges', 'MonthlyCharges']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)

    # 处理所有列中可能存在的NaN值
    df = df.fillna(df.median(numeric_only=True))

    return df


def train_catboost_model(X_train, y_train, X_test, y_test):
    """训练CatBoost模型"""
    # 检查训练数据是否有足够的类别
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(f"训练集中目标变量只有{len(unique_classes)}个类别，无法训练模型。请检查数据。")

    # 明确指定分类特征
    categorical_features = ['PhoneService', 'MultipleLines', 'InternetService',
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']

    # 数值特征
    numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    print(f"分类特征: {categorical_features}")
    print(f"数值特征: {numeric_features}")

    # 获取分类特征的索引
    categorical_features_indices = [X_train.columns.get_loc(col) for col in categorical_features
                                    if col in X_train.columns]

    # 创建CatBoost分类器
    model = CatBoostClassifier(
        iterations=1000,  # 设置为1000轮
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=42,
        verbose=200  # 增加日志频率
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
        plot=False
    )

    print(f"模型训练完成，总迭代次数: {model.tree_count_}")

    return model


def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 分类报告
    report = classification_report(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_proba
    }


def plot_feature_importance(model, feature_names):
    """绘制特征重要性图"""
    feature_importance = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    print(f"所有特征重要性排序:")
    for i, row in feature_importance_df.iterrows():
        print(f"{i + 1}. {row['feature']}: {row['importance']:.4f}")

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance - CatBoost')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('catboost_feature_importance.png')
    plt.show()

    return feature_importance_df


def analyze_churn_reasons(feature_importance_df, model, X_data):
    """分析流失原因"""
    print("=" * 50)
    print("电信用户流失原因分析")
    print("=" * 50)

    # 只分析重要性排名前10的特征
    top_features = feature_importance_df.head(10)['feature'].tolist()

    print("\n1. 最重要的特征（按重要性排序）:")
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"{i + 1}. {row['feature']}: {row['importance']:.4f}")

    print("\n2. 关键特征对流失的影响:")

    for feature in top_features:
        if feature in X_data.columns:
            # 对连续型变量进行处理
            if feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                # 将连续变量分箱
                bins = 5
                labels = [f"第{i + 1}组" for i in range(bins)]
                feature_bins = pd.cut(X_data[feature], bins=bins, labels=labels)

                feature_analysis = pd.DataFrame({
                    'feature_bin': feature_bins,
                    'churn_prob': model.predict_proba(X_data)[:, 1]
                })
                grouped = feature_analysis.groupby('feature_bin', observed=True)['churn_prob'].mean()

                print(f"\n特征 '{feature}':")
                for bin_label, prob in grouped.items():
                    bin_range = X_data[feature][feature_bins == bin_label]
                    if not bin_range.empty:
                        print(f"  区间 [{bin_range.min():.1f}-{bin_range.max():.1f}]: 平均流失概率 = {prob:.3f}")
            else:
                # 分类变量直接分析
                feature_analysis = pd.DataFrame({
                    'feature_value': X_data[feature],
                    'churn_prob': model.predict_proba(X_data)[:, 1]
                })
                grouped = feature_analysis.groupby('feature_value', observed=True)['churn_prob'].mean()

                print(f"\n特征 '{feature}':")
                for value, prob in grouped.items():
                    print(f"  值 {value}: 平均流失概率 = {prob:.3f}")


# 新增: SHAP分析函数
def analyze_with_shap(model, X_train, X_test):
    """使用SHAP进行模型解释"""
    print("\n" + "=" * 50)
    print("SHAP值分析 - 模型解释")
    print("=" * 50)
    
    # 创建SHAP解释器
    # 使用样本数据加速计算，完整数据集可能耗时较长
    sample_size = min(1000, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)
    
    try:
        print("\n正在计算SHAP值...")
        explainer = shap.TreeExplainer(model)
        
        # 在测试集上计算SHAP值
        print("正在计算测试集SHAP值...")
        shap_values = explainer.shap_values(X_test)
        
        # 1. 绘制全局特征重要性总结图
        print("\n绘制SHAP摘要图...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("SHAP全局特征重要性")
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png')
        plt.show()
        
        # 2. 绘制详细的特征影响图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("SHAP特征影响分布")
        plt.tight_layout()
        plt.savefig('shap_feature_impact.png')
        plt.show()
        
        # 3. 分析前5个重要特征的依赖图
        # 获取特征重要性排序
        shap_importance = np.abs(shap_values).mean(0)
        top_indices = shap_importance.argsort()[-5:]  # 获取前5个重要特征的索引
        
        for i in top_indices:
            feature_name = X_test.columns[i]
            print(f"\n分析特征 '{feature_name}' 的SHAP依赖性...")
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(feature_name, shap_values, X_test, show=False)
            plt.title(f"SHAP依赖图 - {feature_name}")
            plt.tight_layout()
            plt.savefig(f'shap_dependence_{feature_name}.png')
            plt.show()
        
        # 4. 分析几个高风险客户样本
        high_risk_indices = np.argsort(model.predict_proba(X_test)[:, 1])[-3:]  # 获取流失概率最高的3个样本
        
        print("\n高风险客户SHAP解释:")
        for i, idx in enumerate(high_risk_indices):
            print(f"\n客户 #{i+1} (流失概率: {model.predict_proba(X_test.iloc[[idx]])[:, 1][0]:.4f}):")
            
            plt.figure(figsize=(12, 6))
            shap.force_plot(
                explainer.expected_value,
                shap_values[idx],
                X_test.iloc[idx],
                matplotlib=True,
                show=False
            )
            plt.title(f"客户 #{i+1} 流失因素分析")
            plt.tight_layout()
            plt.savefig(f'shap_force_plot_customer_{i+1}.png')
            plt.show()
            
            # 打印该客户的主要流失因素
            feature_values = X_test.iloc[idx].to_dict()
            shap_contribution = pd.Series(shap_values[idx], index=X_test.columns)
            top_pos = shap_contribution.nlargest(3)  # 正向贡献最大的特征
            top_neg = shap_contribution.nsmallest(3)  # 负向贡献最大的特征
            
            print("  推动流失的主要因素:")
            for feature, value in top_pos.items():
                print(f"  - {feature}: {feature_values[feature]}, SHAP值: {value:.4f}")
                
            print("  抑制流失的主要因素:")
            for feature, value in top_neg.items():
                print(f"  - {feature}: {feature_values[feature]}, SHAP值: {value:.4f}")
        
        # 5. SHAP交互分析
        print("\n计算SHAP交互效应...")
        # 仅对样本数据计算交互效应以节省时间
        sample_test = X_test.sample(min(200, len(X_test)), random_state=42)
        try:
            shap_interaction = explainer.shap_interaction_values(sample_test)
            
            # 获取最重要的两个特征索引
            feature_idx1, feature_idx2 = np.unravel_index(
                np.abs(shap_interaction).sum(0).sum(0).argmax(), 
                (len(X_test.columns), len(X_test.columns))
            )
            
            feature1 = X_test.columns[feature_idx1]
            feature2 = X_test.columns[feature_idx2]
            
            print(f"\n分析特征 '{feature1}' 和 '{feature2}' 的交互效应...")
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(
                (feature_idx1, feature_idx2),
                shap_interaction,
                sample_test,
                show=False
            )
            plt.title(f"SHAP交互效应: {feature1} × {feature2}")
            plt.tight_layout()
            plt.savefig(f'shap_interaction_{feature1}_{feature2}.png')
            plt.show()
        except Exception as e:
            print(f"SHAP交互分析失败: {e}")
            print("跳过交互分析部分...")
        
    except Exception as e:
        print(f"SHAP分析过程中出错: {e}")
        print("尝试简化的SHAP分析...")
        try:
            # 简化版SHAP分析
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.sample(min(200, len(X_test)), random_state=42))
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test.sample(min(200, len(X_test)), random_state=42), show=False)
            plt.title("SHAP特征影响分布 (简化版)")
            plt.tight_layout()
            plt.savefig('shap_feature_impact_simplified.png')
            plt.show()
        except:
            print("简化的SHAP分析也失败，可能需要检查环境配置或模型兼容性。")


def main():
    fix_matplotlib_display()
    # 读取数据


    try:
        filepath = input("请输入文件名: ")  # 例如: resource/dianxindata.txt
        df = read_file(filepath)
        print(f"成功读取数据，共{len(df)}行")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 数据预处理
    df_processed = preprocess_data(df)

    # 准备特征和目标变量
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    # 检查目标变量的分布
    print("\n最终目标变量分布:")
    unique_counts = y.value_counts()
    print(unique_counts)

    if len(unique_counts) < 2:
        print("错误: 目标变量只有一个唯一值，无法训练模型")
        return

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 验证划分后的数据
    print("\n训练集目标变量分布:")
    print(y_train.value_counts())
    print("测试集目标变量分布:")
    print(y_test.value_counts())

    # 训练CatBoost模型
    try:
        print("\n开始训练CatBoost模型...")
        model = train_catboost_model(X_train, y_train, X_test, y_test)

        # 评估模型
        print("\n评估模型性能...")
        evaluation_results = evaluate_model(model, X_test, y_test)

        # 输出评估结果
        print(f"\n模型评估指标:")
        print(f"准确率 (Accuracy): {evaluation_results['accuracy']:.4f}")
        print(f"精确率 (Precision): {evaluation_results['precision']:.4f}")
        print(f"召回率 (Recall): {evaluation_results['recall']:.4f}")
        print(f"F1分数 (F1-Score): {evaluation_results['f1_score']:.4f}")

        print(f"\n混淆矩阵:")
        print(evaluation_results['confusion_matrix'])

        print(f"\n分类报告:")
        print(evaluation_results['classification_report'])

        # 特征重要性分析
        print("\n分析特征重要性...")
        feature_importance_df = plot_feature_importance(model, X.columns.tolist())

        # 流失原因分析
        analyze_churn_reasons(feature_importance_df, model, X)

        # 预测结果示例
        print(f"\n3. 前10个测试样本的预测结果:")
        sample_predictions = pd.DataFrame({
            '实际值': y_test[:10],
            '预测值': evaluation_results['predictions'][:10],
            '流失概率': evaluation_results['prediction_probabilities'][:10]
        })
        print(sample_predictions)

        # 流失率统计
        churn_rate = y.mean()
        print(f"\n4. 整体流失率: {churn_rate:.3f}")
        print(f"   预测流失率: {evaluation_results['predictions'].mean():.3f}")

        # 新增: SHAP分析
        print("\n5. 使用SHAP进行深入的模型解释...")
        analyze_with_shap(model, X_train, X_test)

    except Exception as e:
        print(f"模型训练或评估过程中出错: {e}")


if __name__ == "__main__":
    main()