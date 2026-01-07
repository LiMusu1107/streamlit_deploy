# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# 设置页面
st.set_page_config(
    page_title="LGG Test App",
    page_icon="🧬",
    layout="wide"
)

# 应用标题
st.title("🧬 LGG Risk Prediction - Compatibility Test")

# 显示版本信息
st.write("### Package Versions")
st.write(f"- Streamlit: {st.__version__}")
st.write(f"- Pandas: {pd.__version__}")
st.write(f"- NumPy: {np.__version__}")
st.write(f"- PyTorch: {torch.__version__}")
st.write(f"- CUDA Available: {torch.cuda.is_available()}")

# 测试基本功能
st.write("### Functionality Tests")

# 测试 NumPy
test_array = np.array([1, 2, 3, 4, 5])
st.write("NumPy array:", test_array)

# 测试 Pandas
df = pd.DataFrame({
    'Sample': ['Patient1', 'Patient2', 'Patient3'],
    'Risk_Score': [0.2, 0.8, 0.5],
    'Group': ['Low', 'High', 'Medium']
})
st.write("Pandas DataFrame:")
st.dataframe(df)

# 测试 Matplotlib
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(df['Sample'], df['Risk_Score'], color=['blue', 'red', 'green'])
ax.set_title("Risk Scores")
ax.set_ylabel("Risk Score")
st.pyplot(fig)

# 测试 PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0, 3.0])
st.write(f"PyTorch tensor on {device}:", x)

st.success("✅ All tests passed! Dependencies are working correctly.")