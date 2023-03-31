import streamlit as st
import openai
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.llms import OpenAIChat
import plotly.express as px
import pandas as pd
import string
import os

os.environ["OPENAI_API_KEY"] = st.secrets['api_key']
llm = OpenAIChat(model_name="gpt-3.5-turbo", temperature=0.0)

PROMPT_TEMPLATE = string.Template('''
データは、PDF を UnstructuredPDFLoader によりテキスト化したものである。
平成24年から令和4年までの行をJSONデータに変換せよ。
出力例は平成24年度の行をJSON化した場合の例である。

# データ:
{$DATA}

# 出力例:
[
  {
    "年": "平成24年",
    "総労働時間": 109.4,
    "所定内労働時間": 109.6,
    "所定外労働時間": 105.6,
  },
]

# JSON:
''')

# function
def convert_pdf_to_opendata(load_data):
    prompt = PROMPT_TEMPLATE.safe_substitute({"DATA": load_data[0].page_content})
    result = llm(prompt)
    data_json = eval(result)
    data_df = pd.DataFrame(data_json)
    return data_json, data_df

def line_graph(df):
    df_tidy = df.melt(id_vars=["年"], value_vars=["総労働時間", "所定内労働時間", "所定外労働時間"], 
                      var_name="労働時間", value_name="労働時間指数(令和２年=100)")
    fig = px.line(df_tidy, x="年", y="労働時間指数(令和２年=100)", color="労働時間")
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    return fig
    
# Layout & Logic
st.title("PDFからオープンデータに開放アプリ")

data_json, data_df = list(), pd.DataFrame()

with st.sidebar:
    pdf_file = st.file_uploader("PDF_DATA：", type={"pdf"})
    if pdf_file is not None:
        loader = UnstructuredPDFLoader(pdf_file)
        load_data = loader.load()

 data_json, data_df = convert_pdf_to_opendata(load_data)

st.dataframe(data_df)
st.plotly_chart(line_graph(data_df))

if not data_df.empty:
    st.sidebar.download_button(
        label="JSONダウンロード",
        data=data_json,
        file_name='労働時間指数.json',
        mime='json',
    )
