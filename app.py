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

if 'data_json' not in st.session_state:
    st.session_state['data_json'] = list()

if 'data_df' not in st.session_state:
    st.session_state['data_df'] = pd.DataFrame()
    
load_data, pdf_file = None, None

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
    data_json = llm(prompt)
    data_lst = eval(data_json)
    data_df = pd.DataFrame(data_lst)
    return data_json, data_df

def line_graph(df):
    df_tidy = df.melt(id_vars=["年"], value_vars=["総労働時間", "所定内労働時間", "所定外労働時間"], 
                      var_name="労働時間", value_name="労働時間指数(令和２年=100)")
    fig = px.line(df_tidy, x="年", y="労働時間指数(令和２年=100)", color="労働時間")
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    return fig
    
# Layout & Logic
st.title("オープンデータ開放アプリ")

pdf_file = st.sidebar.file_uploader("PDFファイル：", type={"pdf"})

if pdf_file:
    try:
        with st.spinner("只今、PDFからオープンデータに開放中・・・　しばらくお待ち下さい。"):
            loader = UnstructuredPDFLoader(pdf_file)
            load_data = loader.load()
            data_json, data_df = convert_pdf_to_opendata(load_data)
            st.dataframe(data_df, height=423)
            st.plotly_chart(line_graph(data_df))
            st.session_state['data_json'] = data_json
            st.session_state['data_df'] = data_df
    except Exception:
        st.error(f'エラーが発生しました。　再度お試し下さい。')
elif not st.session_state['data_df'].empty:
    st.dataframe(st.session_state['data_df'], height=423)
    st.plotly_chart(line_graph(st.session_state['data_df']))

if not st.session_state['data_df'].empty:
    st.sidebar.download_button(
        label="JSONダウンロード",
        data=st.session_state['data_json'],
        file_name='労働時間指数.json',
        mime='text',
    )
