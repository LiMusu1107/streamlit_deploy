import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="加法计算器",
    page_icon="🧮",
    layout="centered"
)

# 应用标题
st.title("🧮 简易加法计算器")
st.markdown("输入两个数字，点击计算按钮即可得到它们的和")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    # 第一个数字输入
    num1 = st.number_input(
        "请输入第一个数字",
        value=0.0,
        format="%.2f"
    )

with col2:
    # 第二个数字输入
    num2 = st.number_input(
        "请输入第二个数字", 
        value=0.0,
        format="%.2f"
    )

# 计算按钮
if st.button("计算总和", type="primary"):
    result = num1 + num2
    st.success(f"### 计算结果: {num1} + {num2} = **{result:.2f}**")
    
    # 添加一些视觉效果
    st.balloons()  # 气球动画
    
    # 显示详细信息
    with st.expander("查看计算详情"):
        st.write(f"**第一个数字**: {num1}")
        st.write(f"**第二个数字**: {num2}")
        st.write(f"**运算**: 加法")
        st.write(f"**结果**: {result}")

# 侧边栏
with st.sidebar:
    st.header("ℹ️ 关于")
    st.info("这是一个简单的加法计算器应用，用于学习 Streamlit Cloud 部署。")
    st.markdown("---")
    st.markdown("**使用方法**:")
    st.markdown("1. 在输入框中输入数字")
    st.markdown("2. 点击'计算总和'按钮")
    st.markdown("3. 查看计算结果")
    
    # 添加一个重置按钮
    if st.button("重置输入"):
        st.rerun()

# 页脚
st.markdown("---")
st.caption("这是一个演示应用，部署在 Streamlit Cloud 上")