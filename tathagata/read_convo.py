import streamlit as st
import json
import re
import pandas as pd

st.set_page_config(page_title="Therapy Chat Viewer", layout="wide")
st.title("Therapy Chat Session")


col1, col2, col3 = st.columns(3)
models = ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'llama3.1:8b', 'llama3.2', 'mistral:7b', 'gemma3:12b', 'gemini-2.5-pro-exp-03-25']
# conv_time = 5

with col1:
    therapist_model = st.selectbox(
        "Select Therapist Model",
        models
    )


with col2:
    patient_model = st.selectbox(
        "Select Patient Model",
        models
    )
    
with col3:
    conv_time = st.selectbox(
        "Select Conversation Time Limit (min)",
        ['5', '10', '15', '20']
    )


filename = f"results/{therapist_model}_{patient_model}_{conv_time}_conversation.json"
try:
    with open(filename, "r") as f:
        raw = f.read()

    fixed_raw = re.sub(r'\]\s*\[', ',', raw)  # replace ][ with ,
    fixed_raw = re.sub(r'\]\s*\{', ',{', fixed_raw)  # replace ]{ with ,
    if not fixed_raw.strip().startswith("["):
        fixed_raw = "[" + fixed_raw
    if not fixed_raw.strip().endswith("]"):
        fixed_raw += "]"
        
    conversation = json.loads(fixed_raw)
        
        
        

    for entry in conversation:
        if "therapist" in entry:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                    <div style="background-color: #e3defa; color: #000; padding: 12px 16px; border-radius: 16px; max-width: 70%; font-size: 16px;">
                        <b style="font-size:20px; color: #40292b;">Therapist</b><br>{entry['therapist']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif "patient" in entry:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                    <div style="background-color: #f4fade; color: #000; padding: 12px 16px; border-radius: 16px; max-width: 70%; font-size: 16px;">
                        <b style="font-size:20px; color: #40292b;">Patient</b><br>{entry['patient']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
except FileNotFoundError:
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Experiment Invalid !!\n\n</p>', unsafe_allow_html=True)
            
df = pd.read_excel('results/summary_metrics.xlsx', index_col=False)
st.write(df)