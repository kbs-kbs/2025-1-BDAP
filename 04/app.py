import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from utils import load_data
from home import run_home
from eda.eda_home import run_eda

def main():
    st.set_page_config(page_title="빅데이터 분석 프로젝트", layout='wide')
    with st.sidebar:
        selected = option_menu(
            "대시보드 메뉴", ["홈", "탐색적 자료분석", "부동산 예측"],
            icons=["house", "bar-chart", "graph-up"],
            menu_icon="cast", default_index=0
        )
        
    total_df = load_data()
        
    if selected == "홈":
        run_home()
    elif selected == "탐색적 자료분석":
        run_eda(total_df)
    elif selected == "부동산 예측":
        st.markdown('부동산 예측입니다.')
    else:
        print("error..")
        
if __name__ == "__main__":
    main()