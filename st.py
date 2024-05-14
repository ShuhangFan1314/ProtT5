import streamlit as st
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import joblib
import torch
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 加载模型和tokenizer
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    model = T5EncoderModel.from_pretrained(model_link).to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(model_link, do_lower_case=False, legacy=True)
    return model, tokenizer

model, tokenizer = load_models()

# 处理序列的函数保持不变...

# Streamlit应用主体
def main():
    st.title("Protein Feature Extraction & Model Evaluation")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(data.head())
        
        # 用户可以选择PCA的组件数量
        n_components = st.slider("Select Number of PCA Components", min_value=1, max_value=32, value=10)
        
        # 特征提取逻辑可以在这里调用，但注意实际应用中可能需要考虑计算资源和时间限制
        
        # 假设我们已经通过某种方式（可能是预计算或简化逻辑）获得了处理后的特征数据
        # X = ...  # 特征数据
        # y = ...  # 目标变量数据
        
        # 如果需要执行PCA和模型训练，这里可以添加逻辑，但请记住实时训练可能不适合Web应用
        
        # 展示PCA结果或模型评估结果（这里以模拟数据为例）
        if st.button("Evaluate Model"):
            # 假设我们有一个预训练好的模型或简化版本的模型评估逻辑
            # 加载预训练模型或结果
            result_path = '../rf_pca_tuning_results/result_component_32.pkl'
            if os.path.exists(result_path):
                results = joblib.load(result_path)
                train_pcc = results.get('Train_PCC', None)
                test_pcc = results.get('Test_PCC', None)
                
                if train_pcc is not None and test_pcc is not None:
                    st.write(f"PCA Components: {n_components}")
                    st.write(f"Training Set Pearson Correlation Coefficient: {train_pcc}")
                    st.write(f"Test Set Pearson Correlation Coefficient: {test_pcc}")
                else:
                    st.warning("No precomputed results found for the specified configuration.")
            else:
                st.error("The specified result file does not exist.")

if __name__ == "__main__":
    main()