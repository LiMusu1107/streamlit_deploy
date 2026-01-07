import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import tempfile
from io import BytesIO
import warnings
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

warnings.filterwarnings('ignore')

# 添加mogonet模块到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mogonet')))

# 导入mogonet模块
try:
    from mogonet.models_FNN import init_model_dict
    from mogonet.train_test_FNN import test_epoch
    from mogonet.utils import load_model_dict_cpu, one_hot_tensor, cal_sample_weight
except ImportError:
    st.error("Error: Could not import mogonet modules. Please ensure the mogonet folder is in the correct location.")
    st.stop()

# 设置页面配置
st.set_page_config(
    page_title="LGG Comprehensive Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("🏥 LGG Comprehensive Prediction System")

# 在侧边栏添加应用说明
st.sidebar.header("📊 About")
st.sidebar.info("""
This application provides two complementary prediction tools for LGG (Low-Grade Glioma) patients:

1. **Survival Prediction**: Predicts survival probability based on clinical parameters
2. **Risk Group Prediction**: Predicts risk group (High/Low) based on multi-omics data

Use the Risk Group Prediction tool if you're unsure of the patient's risk group.
""")

# ============================================================================
# 第一部分：生存预测参数
# ============================================================================
st.sidebar.header("🔧 Survival Prediction Parameters")

# 1. Age slider (20-90)
age = st.sidebar.slider(
    "Age (years)",
    min_value=20,
    max_value=90,
    value=50,
    step=1,
    help="Select patient age between 20 and 90 years"
)

# 2. Grade selection
grade_options = ['G2', 'G3']
grade = st.sidebar.selectbox(
    "Grade",
    options=grade_options,
    index=1,  # Default to G3
    help="Select tumor grade (G2 or G3)"
)

# 3. Risk group selection
risk_options = {
    "High": 1,
    "Low": 0
}
risk_label = st.sidebar.selectbox(
    "Risk Group",
    options=list(risk_options.keys()),
    index=0,  # Default to High
    help="Select risk group (High = 1, Low = 0). Use the Risk Group Prediction tool below if unsure."
)
risk_value = risk_options[risk_label]

# 添加生存预测计算按钮
survival_calc_button = st.sidebar.button(
    "🚀 Calculate Survival Prediction",
    type="primary",
    use_container_width=True
)

# ============================================================================
# 第二部分：风险组别预测参数
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.header("🧬 Risk Group Prediction")

st.sidebar.markdown("""
Use this tool if you're unsure of the patient's risk group.
Upload multi-omics data to predict whether the patient is High or Low risk.
""")

# 在侧边栏添加文件上传区域
st.sidebar.subheader("📤 Upload Omic Data")
st.sidebar.markdown("Please upload CSV files for each omic type (50 features each):")

# 文件上传区域
uploaded_files = {}

col1, col2, col3 = st.sidebar.columns(3)

with col1:
    uploaded_files["mRNA"] = st.file_uploader(
        "mRNA Data", 
        type=["csv"],
        help="Upload mRNA expression data (50 features)",
        key="mrna_uploader"
    )

with col2:
    uploaded_files["miRNA"] = st.file_uploader(
        "miRNA Data", 
        type=["csv"],
        help="Upload miRNA expression data (50 features)",
        key="mirna_uploader"
    )

with col3:
    uploaded_files["methylation"] = st.file_uploader(
        "Methylation Data", 
        type=["csv"],
        help="Upload DNA methylation data (50 features)",
        key="meth_uploader"
    )

# 在侧边栏添加预测方式选择
st.sidebar.subheader("🎯 Prediction Mode")
omic_type = st.sidebar.selectbox(
    "Select Prediction Mode",
    options=["multiomics", "mRNA", "miRNA", "methylation"],
    index=0,
    help="Select the prediction mode"
)

# 显示当前预测模式的要求
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Mode Requirements")

if omic_type == "multiomics":
    st.sidebar.info("**Multiomics Mode**: Requires all three omic data files")
elif omic_type == "mRNA":
    st.sidebar.info("**mRNA Mode**: Requires only mRNA data file")
elif omic_type == "miRNA":
    st.sidebar.info("**miRNA Mode**: Requires only miRNA data file")
elif omic_type == "methylation":
    st.sidebar.info("**Methylation Mode**: Requires only methylation data file")

# 添加风险组别预测按钮
risk_calc_button = st.sidebar.button(
    "🧬 Predict Risk Group",
    type="secondary",
    use_container_width=True
)

# 在侧边栏添加下载示例数据的链接
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Example Data")
st.sidebar.markdown("""
Download example data files for testing:
- [mRNA Example](https://github.com/LiMusu1107/streamlit_deploy/raw/main/data/example_mrna.csv)
- [miRNA Example](https://github.com/LiMusu1107/streamlit_deploy/raw/main/data/example_mirna.csv)
- [Methylation Example](https://github.com/LiMusu1107/streamlit_deploy/raw/main/data/example_meth.csv)
""")

# ============================================================================
# 模型文件夹映射
# ============================================================================
MODEL_MAP = {
    "multiomics": "model_trained/model-early-FNN-multiomics",
    "mRNA": "model_trained/model-early-FNN-mRNA_array",
    "miRNA": "model_trained/model-early-FNN-miRNA",
    "methylation": "model_trained/model-early-FNN-methy"
}

# 预期特征维度
EXPECTED_DIMS = {
    "multiomics": 150,  # 3 * 50 features
    "mRNA": 50,
    "miRNA": 50,
    "methylation": 50
}

# ============================================================================
# 风险组别预测相关函数
# ============================================================================
@st.cache_data
def load_training_data():
    try:
        # 加载训练索引
        train_idx = pd.read_csv("data/train_index.csv")
        train_idx = train_idx.values.flatten() - 1
        
        # 加载标签
        df_label = pd.read_csv("data/tcga_label2.csv")
        df_label = df_label.rename(columns={df_label.columns[1]: 'label'})
        label = df_label['label'].values - 1
        
        return train_idx, label
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return None, None

# 初始化模型
def init_and_load_model(model_folder, dim_list):
    try:
        # 模型结构参数
        view_list = [1]
        num_view = len(view_list)
        num_class = 2
        dim_hvcdn = 100
        dim_he_list = [300, 200, 100]
        dropout_rate = 0.5
        
        # 初始化模型
        model = init_model_dict(
            num_view, num_class, dim_list, dim_he_list, dim_hvcdn, dropout_rate
        )
        
        # 加载训练好的模型
        return load_model_dict_cpu(model_folder, model)
    except Exception as e:
        st.error(f"Error initializing or loading model: {str(e)}")
        return None

# 处理上传的文件
def process_uploaded_file(uploaded_file, expected_dim=50):
    try:
        # 使用临时文件处理上传
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # 读取CSV文件
        df = pd.read_csv(tmp_file_path)
        
        # 检查维度
        if df.shape[1] != expected_dim:
            st.error(f"Invalid dimension. Expected {expected_dim} features, got {df.shape[1]}.")
            return None, f"Expected {expected_dim} features, got {df.shape[1]}"
        
        # 删除临时文件
        os.unlink(tmp_file_path)
        
        return df.to_numpy(), None
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None, str(e)

# 检查是否所有必需的文件都已上传
def check_required_files(omic_type, uploaded_files):
    required_files = []
    
    if omic_type == "multiomics":
        required_files = ["mRNA", "miRNA", "methylation"]
    elif omic_type == "mRNA":
        required_files = ["mRNA"]
    elif omic_type == "miRNA":
        required_files = ["miRNA"]
    elif omic_type == "methylation":
        required_files = ["methylation"]
    
    missing_files = []
    for file_type in required_files:
        if file_type not in uploaded_files or uploaded_files[file_type] is None:
            missing_files.append(file_type)
    
    return missing_files

# ============================================================================
# 生存预测相关函数
# ============================================================================
def perform_survival_prediction(age, grade, risk_value, risk_label):
    with st.spinner("🔬 Loading data and training model..."):
        # 1. 读取数据
        data_tcga = pd.read_csv("survdata_tcga_lgg.csv")
        
        # 2. 转换分类变量
        categorical_cols = ['label2', 'grade', 'histological_type', 'IDH']
        for col in categorical_cols:
            if col in data_tcga.columns:
                data_tcga[col] = data_tcga[col].astype('category')
        
        # 3. 准备用于Cox模型的数据
        data_for_cox = data_tcga[['os', 'censor', 'age', 'grade', 'label2']].copy()
        data_for_cox = data_for_cox.dropna()
        
        # 4. 拟合Cox比例风险模型
        cph = CoxPHFitter()
        cph.fit(data_for_cox, duration_col='os', event_col='censor', 
                formula='age + grade + label2')
        
        # 5. 根据用户输入创建新患者数据
        new_patient = pd.DataFrame({
            'age': [age],
            'grade': pd.Categorical([grade]),
            'label2': pd.Categorical([risk_value])
        })
        
        # 6. 预测生存函数
        survival_function = cph.predict_survival_function(new_patient)
        
        # 转换为DataFrame
        survival_df = pd.DataFrame({
            'time': survival_function.index,
            'surv': survival_function.iloc[:, 0]
        })
        
        # 7. 计算特定时间点的生存率
        time_points = [12, 36, 60, 84, 108, 120]  # 1,3,5,7,9,10年
        time_labels = ['1 year', '3 years', '5 years', '7 years', '9 years', '10 years']
        
        results = []
        for t, label in zip(time_points, time_labels):
            idx = (survival_df['time'] - t).abs().idxmin()
            if idx < len(survival_df):
                surv_prob = survival_df.loc[idx, 'surv']
                se = surv_prob * (1 - surv_prob) / np.sqrt(len(data_for_cox))
                ci_lower = max(0, surv_prob - 1.96 * se)
                ci_upper = min(1, surv_prob + 1.96 * se)
                
                results.append({
                    'Time': label,
                    'Survival Rate (%)': f"{surv_prob * 100:.2f}%",
                    '95% CI': f"[{ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%]"
                })
        
        survival_results = pd.DataFrame(results)
        
        # 8. 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制生存曲线
        ax1.step(survival_df['time'], survival_df['surv'], 
                 linewidth=2, color='#25558F', where='post')
        ax1.fill_between(survival_df['time'], 
                          survival_df['surv'] * 0.9,
                          survival_df['surv'] * 1.1,
                          alpha=0.2, color='#25558F')
        
        # 标记特定的时间点
        for t, label in zip(time_points, time_labels):
            idx = (survival_df['time'] - t).abs().idxmin()
            if idx < len(survival_df):
                ax1.scatter(t, survival_df.loc[idx, 'surv'], 
                           color='red', s=50, zorder=5)
                ax1.text(t, survival_df.loc[idx, 'surv'] + 0.05, 
                        f'{label}\n{survival_df.loc[idx, "surv"]:.1%}',
                        ha='center', fontsize=10)
        
        ax1.set_xlabel('Survival Time (months)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax1.set_title(f'Predicted Survival Curve for {age}-year-old patient\n'
                     f'Grade: {grade}, Risk Group: {risk_label}', 
                     fontsize=14, fontweight='bold', color='#25558F')
        ax1.set_xlim([0, 125])
        ax1.set_ylim([0, 1.05])
        ax1.set_xticks(range(0, 121, 12))
        ax1.set_yticks(np.arange(0, 1.1, 0.2))
        ax1.tick_params(axis='both', labelsize=10)
        
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 添加表格
        ax2.axis('tight')
        ax2.axis('off')
        
        table = ax2.table(cellText=survival_results.values,
                         colLabels=survival_results.columns,
                         cellLoc='center',
                         loc='center',
                         colColours=['#f0f0f0']*len(survival_results.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        plt.tight_layout()
        
        return fig, survival_df, survival_results, cph, data_for_cox

# ============================================================================
# 主应用逻辑
# ============================================================================
# 初始化session state
if 'risk_prediction_result' not in st.session_state:
    st.session_state.risk_prediction_result = None
if 'last_risk_value' not in st.session_state:
    st.session_state.last_risk_value = None
if 'last_risk_label' not in st.session_state:
    st.session_state.last_risk_label = None

# 主内容区域
# 如果没有任何计算，显示欢迎信息
if not survival_calc_button and not risk_calc_button:
    st.markdown("""
    ## Welcome to LGG Comprehensive Prediction System
    
    This application provides two complementary prediction tools for LGG (Low-Grade Glioma) patients:
    
    ### 1. Survival Prediction
    Predicts survival probability based on clinical parameters:
    - **Age**: Patient age in years
    - **Grade**: Tumor grade (G2/G3)
    - **Risk Group**: High (1) or Low (0) risk
    
    ### 2. Risk Group Prediction
    Predicts risk group (High/Low) based on multi-omics data when you're unsure of the risk group.
    - Supports multiomics, mRNA, miRNA, and methylation data
    - Each omic data file should have exactly 50 features
    
    ### How to use:
    1. **Set survival prediction parameters** in the first section of the sidebar
    2. **Optionally, use risk group prediction** in the second section if unsure of risk group
    3. **Click the respective calculation buttons**
    4. **View results** in the main area
    
    ⬅️ **Please set parameters in the sidebar and click the calculation buttons.**
    """)
    
    # 显示示例图片占位符
    st.info("👈 Set parameters in the sidebar and click calculation buttons to see results.")

# 处理风险组别预测
if risk_calc_button:
    st.subheader("🧬 Risk Group Prediction Results")
    
    # 检查是否所有必需的文件都已上传
    missing_files = check_required_files(omic_type, uploaded_files)
    
    if missing_files:
        st.error(f"Missing required files for {omic_type} prediction: {', '.join(missing_files)}")
        st.info("Please upload the missing files in the sidebar.")
    else:
        # 处理上传的文件
        with st.spinner("Processing uploaded data..."):
            data_arrays = {}
            errors = []
            
            # 处理所有上传的文件
            for file_type, uploaded_file in uploaded_files.items():
                if uploaded_file is not None:
                    data_array, error = process_uploaded_file(uploaded_file, expected_dim=50)
                    if error:
                        errors.append(f"{file_type}: {error}")
                    elif data_array is not None:
                        data_arrays[file_type] = data_array
            
            # 检查是否有处理错误
            if errors:
                st.error("Errors processing uploaded files:")
                for error in errors:
                    st.error(f"- {error}")
            else:
                # 根据预测模式准备测试数据
                if omic_type == "multiomics":
                    # 检查所有三个组学数据是否都已上传
                    required_omics = ["mRNA", "miRNA", "methylation"]
                    missing_omics = [omic for omic in required_omics if omic not in data_arrays]
                    
                    if missing_omics:
                        st.error(f"Missing omic data for multiomics prediction: {', '.join(missing_omics)}")
                    else:
                        # 检查所有组学数据是否有相同的样本数
                        sample_counts = {omic: data.shape[0] for omic, data in data_arrays.items()}
                        if len(set(sample_counts.values())) > 1:
                            st.error(f"Inconsistent sample counts: {sample_counts}")
                        else:
                            # 拼接多组学数据
                            test_X = np.concatenate(
                                [data_arrays["mRNA"], data_arrays["miRNA"], data_arrays["methylation"]], 
                                axis=1
                            )
                            
                            st.success(f"Multiomics data prepared: {test_X.shape[0]} samples, {test_X.shape[1]} features")
                else:
                    # 单组学预测
                    if omic_type not in data_arrays:
                        st.error(f"No {omic_type} data uploaded for {omic_type} prediction.")
                    else:
                        test_X = data_arrays[omic_type]
                        st.success(f"{omic_type} data prepared: {test_X.shape[0]} samples, {test_X.shape[1]} features")
                
                # 如果数据准备成功，继续加载训练数据和进行预测
                if 'test_X' in locals():
                    # 加载训练数据
                    with st.spinner("Loading training data references..."):
                        train_idx, label = load_training_data()
                        if train_idx is None or label is None:
                            st.error("Failed to load training data.")
                        else:
                            # 根据组学类型准备训练数据
                            if omic_type == "multiomics":
                                # 加载多组学训练数据
                                try:
                                    omics1 = pd.read_csv("data/tcga_mrna.csv").to_numpy()
                                    omics2 = pd.read_csv("data/tcga_mirna.csv").to_numpy()
                                    omics3 = pd.read_csv("data/tcga_meth.csv").to_numpy()
                                    omics = np.concatenate((omics1, omics2, omics3), axis=1)
                                    train_X = omics[train_idx]
                                except FileNotFoundError:
                                    st.error("Training data files not found. Please ensure the 50-feature training data files are available.")
                                    st.info("Required files: tcga_mrna.csv, tcga_mirna.csv, tcga_meth.csv")
                            elif omic_type == "mRNA":
                                try:
                                    omics1 = pd.read_csv("data/tcga_mrna.csv").to_numpy()
                                    train_X = omics1[train_idx]
                                except FileNotFoundError:
                                    st.error("mRNA training data file not found: tcga_mrna.csv")
                            elif omic_type == "miRNA":
                                try:
                                    omics2 = pd.read_csv("data/tcga_mirna.csv").to_numpy()
                                    train_X = omics2[train_idx]
                                except FileNotFoundError:
                                    st.error("miRNA training data file not found: tcga_mirna.csv")
                            elif omic_type == "methylation":
                                try:
                                    omics3 = pd.read_csv("data/tcga_meth.csv").to_numpy()
                                    train_X = omics3[train_idx]
                                except FileNotFoundError:
                                    st.error("Methylation training data file not found: tcga_meth.csv")
                            
                            if 'train_X' in locals():
                                train_y = label[train_idx]
                                test_y = np.zeros(test_X.shape[0], dtype=int)
                                
                                st.info(f"Training data loaded: {train_X.shape[0]} samples, {train_X.shape[1]} features")
                                
                                # 准备预测数据
                                with st.spinner("Preparing data for prediction..."):
                                    # 转换为张量
                                    data_tr_list = [torch.FloatTensor(train_X)]
                                    data_trte_list = [torch.FloatTensor(np.concatenate((train_X, test_X), axis=0))]
                                    
                                    # 强制使用CPU
                                    cuda = False
                                    
                                    # 准备索引
                                    num_tr = data_tr_list[0].shape[0]
                                    num_trte = data_trte_list[0].shape[0]
                                    labels_trte = np.concatenate((train_y, test_y))
                                    trte_idx = {"tr": list(range(num_tr)), "te": list(range(num_tr, num_trte))}
                                    
                                    # 准备标签张量
                                    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
                                    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, 2)
                                    sample_weight_tr = torch.FloatTensor(
                                        cal_sample_weight(labels_trte[trte_idx["tr"]], 2)
                                    )
                                
                                # 加载模型并进行预测
                                with st.spinner(f"Loading {omic_type} model and making predictions..."):
                                    # 获取模型文件夹
                                    model_folder = MODEL_MAP[omic_type]
                                    
                                    # 获取输入维度
                                    dim_list = [x.shape[1] for x in data_tr_list]
                                    
                                    # 初始化并加载模型
                                    trained_model = init_and_load_model(model_folder, dim_list)
                                    if trained_model is None:
                                        st.error(f"Failed to load {omic_type} model from {model_folder}")
                                        st.info("Please ensure the model files are available in the correct directory.")
                                    else:
                                        # 进行预测
                                        try:
                                            predictions = test_epoch(data_trte_list, trte_idx["te"], trained_model)
                                            y_pred = np.argmax(predictions, axis=1)
                                            
                                            # 保存预测结果到session state
                                            st.session_state.risk_prediction_result = {
                                                'predictions': predictions,
                                                'y_pred': y_pred,
                                                'test_X': test_X,
                                                'omic_type': omic_type,
                                                'data_arrays': data_arrays
                                            }
                                            
                                            # 显示上传的数据摘要
                                            with st.expander("📊 Uploaded Data Summary", expanded=True):
                                                st.markdown(f"**Prediction Mode:** {omic_type}")
                                                st.markdown(f"**Number of Samples:** {test_X.shape[0]}")
                                                st.markdown(f"**Number of Features:** {test_X.shape[1]}")
                                                
                                                if omic_type == "multiomics":
                                                    st.markdown("**Features per Omic Type:** 50 (each)")
                                                    st.markdown("**Total Features:** 150 (50 mRNA + 50 miRNA + 50 methylation)")
                                                else:
                                                    st.markdown(f"**Features:** 50")
                                            
                                            st.subheader("🎯 Prediction Results")
                                            
                                            # 创建结果表格
                                            results = []
                                            for i in range(len(y_pred)):
                                                risk_group = "High Risk" if y_pred[i] == 1 else "Low Risk"
                                                high_risk_prob = predictions[i][1] * 100
                                                low_risk_prob = predictions[i][0] * 100
                                                
                                                # 保存第一个样本的结果到session state
                                                if i == 0:
                                                    st.session_state.last_risk_value = 1 if y_pred[i] == 1 else 0
                                                    st.session_state.last_risk_label = risk_group
                                                
                                                # 添加风险解释
                                                if y_pred[i] == 1:
                                                    risk_explanation = "Higher likelihood of disease progression"
                                                else:
                                                    risk_explanation = "Lower likelihood of disease progression"
                                                
                                                results.append({
                                                    "Sample": f"Patient {i+1}",
                                                    "Risk Group": risk_group,
                                                    "High Risk Probability": f"{high_risk_prob:.2f}%",
                                                    "Low Risk Probability": f"{low_risk_prob:.2f}%",
                                                    "Interpretation": risk_explanation
                                                })
                                            
                                            results_df = pd.DataFrame(results)
                                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                                            
                                            # 添加统计信息
                                            high_risk_count = sum(y_pred == 1)
                                            low_risk_count = sum(y_pred == 0)
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("High Risk Patients", f"{high_risk_count} ({high_risk_count/len(y_pred)*100:.1f}%)")
                                            with col2:
                                                st.metric("Low Risk Patients", f"{low_risk_count} ({low_risk_count/len(y_pred)*100:.1f}%)")
                                            
                                            # 如果预测了风险组别，更新侧边栏的风险组别选择
                                            if len(y_pred) > 0:
                                                st.info(f"""
                                                💡 **Suggestion**: The predicted risk group for the first patient is **{st.session_state.last_risk_label}**. 
                                                This value has been automatically selected in the Risk Group dropdown in the sidebar.
                                                You can now use this value for survival prediction.
                                                """)
                                            
                                            # 添加下载按钮
                                            csv = results_df.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download Prediction Results (CSV)",
                                                data=csv,
                                                file_name=f"lgg_{omic_type}_risk_predictions.csv",
                                                mime="text/csv",
                                            )
                                            
                                            # 显示详细预测信息
                                            with st.expander("🔍 Model and Data Details", expanded=False):
                                                st.markdown(f"""
                                                ### Model Information
                                                - **Model Type**: Feedforward Neural Network (FNN)
                                                - **Prediction Mode**: {omic_type}
                                                - **Model Location**: {MODEL_MAP[omic_type]}
                                                - **Training Samples**: {train_X.shape[0]}
                                                - **Input Features**: {train_X.shape[1]}
                                                
                                                ### Data Dimensions
                                                | Data Type | Samples | Features |
                                                |-----------|---------|----------|
                                                | Training Data | {train_X.shape[0]} | {train_X.shape[1]} |
                                                | Test Data | {test_X.shape[0]} | {test_X.shape[1]} |
                                                
                                                ### Uploaded Files Status
                                                | File Type | Status | Features |
                                                |-----------|--------|----------|
                                                | mRNA | {'✅ Uploaded' if 'mRNA' in data_arrays else '❌ Not uploaded'} | {data_arrays['mRNA'].shape[1] if 'mRNA' in data_arrays else 'N/A'} |
                                                | miRNA | {'✅ Uploaded' if 'miRNA' in data_arrays else '❌ Not uploaded'} | {data_arrays['miRNA'].shape[1] if 'miRNA' in data_arrays else 'N/A'} |
                                                | Methylation | {'✅ Uploaded' if 'methylation' in data_arrays else '❌ Not uploaded'} | {data_arrays['methylation'].shape[1] if 'methylation' in data_arrays else 'N/A'} |
                                                """)
                                            
                                        except Exception as e:
                                            st.error(f"Error during prediction: {str(e)}")
                                            st.info("This may be due to dimension mismatch between the model and data.")

# 处理生存预测
if survival_calc_button:
    st.subheader("🏥 Survival Prediction Results")
    
    # 如果最近有风险组别预测结果，显示提示
    if st.session_state.last_risk_label and st.session_state.last_risk_value:
        if st.session_state.last_risk_value != risk_value:
            st.info(f"""
            💡 **Note**: You recently predicted a risk group of **{st.session_state.last_risk_label}** 
            using the Risk Group Prediction tool. The current survival prediction is using the manually selected 
            risk group **{risk_label}**. You may want to update the risk group selection to use the predicted value.
            """)
    
    # 执行生存预测
    try:
        fig, survival_df, survival_results, cph, data_for_cox = perform_survival_prediction(
            age, grade, risk_value, risk_label
        )
        
        # 在Streamlit中显示
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Patient Information")
            st.info(f"""
            **Age:** {age} years  
            **Grade:** {grade}  
            **Risk Group:** {risk_label} (value: {risk_value})
            """)
            
            st.subheader("📈 Survival Rates")
            st.dataframe(
                survival_results,
                use_container_width=True,
                hide_index=True
            )
            
            # 提供数据下载
            csv = survival_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Survival Data (CSV)",
                data=csv,
                file_name=f"survival_prediction_age{age}_grade{grade}_risk{risk_label}.csv",
                mime="text/csv",
            )
        
        with col2:
            st.subheader("📊 Survival Curve")
            st.pyplot(fig)
            
            # 提供图表下载
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            
            st.download_button(
                label="🖼️ Download Chart (PNG)",
                data=buf,
                file_name=f"survival_curve_age{age}_grade{grade}_risk{risk_label}.png",
                mime="image/png",
            )
        
        # 显示模型信息
        with st.expander("📊 Model Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Summary**")
                st.write(f"- Number of patients in training: {len(data_for_cox)}")
                st.write(f"- Features used: Age, Grade, Risk Group")
                st.write(f"- Model: Cox Proportional Hazards")
                st.write(f"- Concordance Index: {cph.concordance_index_:.3f}")
            
            with col2:
                st.write("**Cox Model Coefficients**")
                coef_df = pd.DataFrame({
                    'Feature': cph.params_.index,
                    'Coefficient': cph.params_.values,
                    'Hazard Ratio': np.exp(cph.params_.values)
                })
                st.dataframe(coef_df, use_container_width=True, hide_index=True)
        
        # 添加解释说明
        st.markdown("---")
        st.markdown("""
        ### 📖 Interpretation Guide
        
        1. **Survival Curve**: Shows the probability of survival over time (in months)
        2. **Survival Rate**: Percentage of patients expected to survive at each time point
        3. **95% CI**: 95% confidence interval for the survival estimate
        4. **Grade**: G2 (low grade) vs G3 (high grade) tumors
        5. **Risk Group**: Based on molecular markers (High=1, Low=0)
        
        **Note**: The model is trained on TCGA-LGG data. Predictions are estimates and should be used in conjunction with clinical judgment.
        """)
        
    except Exception as e:
        st.error(f"Error during survival prediction: {str(e)}")
        st.info("Please check if the data files are available and in the correct format.")

# 如果两个计算都执行了，显示分隔线
if survival_calc_button and risk_calc_button:
    st.markdown("---")
    st.subheader("📊 Combined Results Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Risk Group", f"{risk_label}")
        st.metric("Patient Age", f"{age} years")
    
    with col2:
        st.metric("Tumor Grade", grade)
        if survival_calc_button and 'survival_results' in locals():
            # 获取5年生存率
            five_year_survival = None
            for idx, row in survival_results.iterrows():
                if "5 year" in row['Time']:
                    five_year_survival = row['Survival Rate (%)']
                    break
            
            if five_year_survival:
                st.metric("5-Year Survival Rate", five_year_survival)