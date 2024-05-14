import streamlit as st
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib
import torch
import re

# 加载模型和工具
@st.cache(allow_output_mutation=True)
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False, legacy=True)
    scaler = StandardScaler()
    kpca = KernelPCA()  # 初始化PCA，如果要使用预训练的PCA模型，则从文件加载
    # 加载本地保存的随机森林模型
    rf_model = joblib.load('random_forest_model.pkl')
    return model, tokenizer, scaler, kpca, rf_model

model, tokenizer, scaler, kpca, rf_model = load_resources()

def process_single_sequence(seq, model, tokenizer):
    seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
    ids = tokenizer.encode_plus(seq, add_special_tokens=True, padding="max_length", max_length=512, truncation=True)
    input_ids = torch.tensor(ids['input_ids']).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(ids['attention_mask']).unsqueeze(0).to(model.device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    return embedding_repr.last_hidden_state.squeeze().mean(dim=0).tolist()

st.title("基于蛋白质序列的粘度预测")

sequence_input = st.text_area("请输入蛋白质序列")

if sequence_input:
    # 处理序列
    features = process_single_sequence(sequence_input, model, tokenizer)
    # 特征缩放
    scaled_features = scaler.transform([features])
    # 使用PCA变换
    transformed_features = kpca.transform(scaled_features)
    # 使用随机森林模型进行预测
    predicted_viscosity = rf_model.predict(transformed_features)
    
    st.write(f"预测的粘度值为: {predicted_viscosity[0]}")
