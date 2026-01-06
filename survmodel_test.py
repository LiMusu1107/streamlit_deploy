import streamlit as st
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="LGG Patient Survival Prediction",
    page_icon="🏥",
    layout="wide"
)

# 应用标题
st.title("🏥 LGG Patient Survival Prediction")

# 创建侧边栏
st.sidebar.header("🔧 Patient Parameters")

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
    help="Select risk group (High = 1, Low = 0)"
)
risk_value = risk_options[risk_label]

# 4. 添加一个计算按钮
calculate = st.sidebar.button(
    "🚀 Calculate Survival Prediction",
    type="primary",
    use_container_width=True
)

# 5. 在侧边栏添加说明
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info(
    """
    This tool predicts survival probability for LGG (Low-Grade Glioma) patients 
    using a Cox Proportional Hazards model.
    
    **Parameters:**
    - **Age**: Patient age in years
    - **Grade**: Tumor grade (G2/G3)
    - **Risk Group**: High (1) or Low (0) risk
    """
)

# 主内容区域
if calculate or 'autocalc' in st.session_state:
    # 设置标志，确保首次加载也运行
    if 'autocalc' not in st.session_state:
        st.session_state.autocalc = True
    
    # 显示加载状态
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
        
        # 9. 在Streamlit中显示
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
    
    # 显示模型信息 - 修复这里的错误
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
            # 修复：使用 params_ 而不是 params
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
else:
    # 初始状态显示说明
    st.markdown("""
    ## Welcome to LGG Survival Prediction Tool
    
    This application predicts survival probabilities for patients with Low-Grade Glioma (LGG) 
    based on clinical parameters.
    
    ### How to use:
    1. **Adjust parameters** in the left sidebar
    2. **Click the 'Calculate Survival Prediction' button**
    3. **View results** including survival curve and survival rates
    
    ### Default Parameters:
    - **Age**: 50 years
    - **Grade**: G3
    - **Risk Group**: High
    
    ⬅️ **Please adjust the parameters in the sidebar and click the calculate button.**
    """)
    
    # 显示示例图片占位符
    st.info("👈 Adjust the parameters in the sidebar and click 'Calculate Survival Prediction' to see the results.")
