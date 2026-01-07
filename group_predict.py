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
    page_title="LGG Multi-omics Risk Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("🧬 LGG Multi-omics Risk Prediction System")

# 在侧边栏添加应用说明
st.sidebar.header("📊 About")
st.sidebar.info("""
This application predicts risk groups for LGG patients 
using multi-omics data and deep learning models.

**Supported Omic Types:**
- **Multiomics**: Combined mRNA, miRNA and methylation data
- **mRNA**: mRNA expression data only
- **miRNA**: miRNA expression data only
- **Methylation**: DNA methylation data only

**Data Requirements:**
- CSV format with samples as rows and features as columns
- No index column (first column should be data)
- Each omic data should have exactly 50 features
- For multiomics prediction, all three omic types are required
""")

# 在侧边栏添加下载示例数据的链接
st.sidebar.markdown("---")
st.sidebar.header("📁 Example Data")
st.sidebar.markdown("""
Download example data files for testing:
- [mRNA Example](https://github.com/LiMusu1107/streamlit_deploy/raw/main/data/example_mrna.csv)
- [miRNA Example](https://github.com/LiMusu1107/streamlit_deploy/raw/main/data/example_mirna.csv)
- [Methylation Example](https://github.com/LiMusu1107/streamlit_deploy/raw/main/data/example_meth.csv)
""")

# 模型文件夹映射
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

# 加载训练索引和标签
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

# 主应用逻辑
def main():
    # 在侧边栏添加文件上传区域
    st.sidebar.header("📤 Upload Omic Data")
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
    
    # 显示上传状态
    upload_status = {}
    for file_type, uploaded_file in uploaded_files.items():
        if uploaded_file is not None:
            upload_status[file_type] = "✅ Uploaded"
        else:
            upload_status[file_type] = "❌ Not uploaded"
    
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Upload Status")
    for file_type, status in upload_status.items():
        st.sidebar.write(f"**{file_type}**: {status}")
    
    # 在侧边栏添加预测方式选择
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Prediction Mode")
    omic_type = st.sidebar.selectbox(
        "Select Prediction Mode",
        options=["multiomics", "mRNA", "miRNA", "methylation"],
        index=0,
        help="Select the prediction mode"
    )
    
    # 显示当前预测模式的要求
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Mode Requirements")
    
    if omic_type == "multiomics":
        st.sidebar.info("**Multiomics Mode**: Requires all three omic data files")
    elif omic_type == "mRNA":
        st.sidebar.info("**mRNA Mode**: Requires only mRNA data file")
    elif omic_type == "miRNA":
        st.sidebar.info("**miRNA Mode**: Requires only miRNA data file")
    elif omic_type == "methylation":
        st.sidebar.info("**Methylation Mode**: Requires only methylation data file")
    
    # 添加预测按钮
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button(
        "🚀 Predict Risk Group",
        type="primary",
        use_container_width=True
    )
    
    # 主内容区域
    if not predict_button:
        st.markdown("""
        ## Welcome to LGG Risk Prediction System
        
        This application predicts risk groups for Low-Grade Glioma (LGG) patients 
        using multi-omics data and deep learning models.
        
        ### How to use:
        1. **Upload omic data files** in the left sidebar (optional, depending on prediction mode)
        2. **Select prediction mode** (multiomics, mRNA, miRNA, or methylation)
        3. **Click the 'Predict Risk Group' button**
        4. **View prediction results** including risk group and probabilities
        
        ### Upload Guidelines:
        - Each omic data file should have exactly 50 features
        - Samples as rows, features as columns
        - No header row (first row should be data)
        - No index column
        
        ⬅️ **Please upload data and select prediction mode in the sidebar.**
        """)
        
        return
    
    # 检查是否所有必需的文件都已上传
    missing_files = check_required_files(omic_type, uploaded_files)
    
    if missing_files:
        st.error(f"Missing required files for {omic_type} prediction: {', '.join(missing_files)}")
        st.info("Please upload the missing files in the sidebar.")
        return
    
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
            return
        
        # 根据预测模式准备测试数据
        if omic_type == "multiomics":
            # 检查所有三个组学数据是否都已上传
            required_omics = ["mRNA", "miRNA", "methylation"]
            missing_omics = [omic for omic in required_omics if omic not in data_arrays]
            
            if missing_omics:
                st.error(f"Missing omic data for multiomics prediction: {', '.join(missing_omics)}")
                return
            
            # 检查所有组学数据是否有相同的样本数
            sample_counts = {omic: data.shape[0] for omic, data in data_arrays.items()}
            if len(set(sample_counts.values())) > 1:
                st.error(f"Inconsistent sample counts: {sample_counts}")
                return
            
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
                return
            
            test_X = data_arrays[omic_type]
            st.success(f"{omic_type} data prepared: {test_X.shape[0]} samples, {test_X.shape[1]} features")
    
    # 加载训练数据
    with st.spinner("Loading training data references..."):
        train_idx, label = load_training_data()
        if train_idx is None or label is None:
            return
        
        # 根据组学类型准备训练数据
        if omic_type == "multiomics":
            # 加载多组学训练数据
            try:
                omics1 = pd.read_csv("data/tcga_mrna.csv").to_numpy()  # 假设有50维的训练数据
                omics2 = pd.read_csv("data/tcga_mirna.csv").to_numpy()
                omics3 = pd.read_csv("data/tcga_meth.csv").to_numpy()
                omics = np.concatenate((omics1, omics2, omics3), axis=1)
                train_X = omics[train_idx]
            except FileNotFoundError:
                st.error("Training data files not found. Please ensure the 50-feature training data files are available.")
                st.info("Required files: tcga_mrna.csv, tcga_mirna.csv, tcga_meth.csv")
                return
                
        elif omic_type == "mRNA":
            try:
                omics1 = pd.read_csv("data/tcga_mrna.csv").to_numpy()
                train_X = omics1[train_idx]
            except FileNotFoundError:
                st.error("mRNA training data file not found: tcga_mrna.csv")
                return
                
        elif omic_type == "miRNA":
            try:
                omics2 = pd.read_csv("data/tcga_mirna.csv").to_numpy()
                train_X = omics2[train_idx]
            except FileNotFoundError:
                st.error("miRNA training data file not found: tcga_mirna.csv")
                return
                
        elif omic_type == "methylation":
            try:
                omics3 = pd.read_csv("data/tcga_meth.csv").to_numpy()
                train_X = omics3[train_idx]
            except FileNotFoundError:
                st.error("Methylation training data file not found: tcga_meth.csv")
                return
        
        train_y = label[train_idx]
        test_y = np.zeros(test_X.shape[0], dtype=int)
        
        st.info(f"Training data loaded: {train_X.shape[0]} samples, {train_X.shape[1]} features")
    
    # 准备预测数据
    with st.spinner("Preparing data for prediction..."):
        # 转换为张量
        data_tr_list = [torch.FloatTensor(train_X)]
        data_trte_list = [torch.FloatTensor(np.concatenate((train_X, test_X), axis=0))]
        
        # 检查CUDA可用性
        cuda = torch.cuda.is_available()
        if cuda:
            data_tr_list[0] = data_tr_list[0].cuda()
            data_trte_list[0] = data_trte_list[0].cuda()
        
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
        
        if cuda:
            labels_tr_tensor = labels_tr_tensor.cuda()
            onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
            sample_weight_tr = sample_weight_tr.cuda()
    
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
            return
        
        # 进行预测
        try:
            predictions = test_epoch(data_trte_list, trte_idx["te"], trained_model)
            y_pred = np.argmax(predictions, axis=1)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("This may be due to dimension mismatch between the model and data.")
            return
    
    # 显示结果
    st.success("✅ Prediction completed successfully!")
    
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

if __name__ == "__main__":

    main()


