import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from eda.viz import showViz
from eda.stat import showStat
from eda.map import showMap

def home():
    st.markdown("""### Visualization 개요 \n
                - 가구당 평균 가격 추세 \n
                - 가구당 거래 건수 추세 \n
                - 지역별 평균 가격 막대 그래프 \n
                """)
    st.markdown('### Statistics 개요 \n')
    st.markdown('### Map 개요 \n')
    
def run_eda(total_df):
    total_df['CTRT_DAY'] = pd.to_datetime(total_df['CTRT_DAY'], format="%Y-%m-%d")
    
    st.markdown("## 탐색적 자료 분석 개요")
    st.markdown("탐색적 자료 분석 페이지입니다.")
    st.markdown("아래에 넣고 싶은 내용을 추가할 수 있습니다.")
    
    selected = option_menu(
        None, ['Home', 'Visualization', 'Statistics', 'Map'],
        icons=['house', 'bar-chart', 'file-spreadsheet', 'map'],
        menu_icon='cast', default_index=0, orientation='horizontal',
        styles={
            'container': {'padding': '0!important', 'background-color': '#fafafa'},
            'icon': {'color': 'orange', 'font-size': '25px'},
            'nav-link': {'font-size': '18px', 'text-align': 'left', 'margin': '0px',
                         '--hover-color': '#eee'},
            'nav-link-selected': {'background-color': 'green'}
        }
    )

    if selected == 'Home':
        home()
    elif selected == 'Visualization':
        showViz(total_df)
    elif selected == 'Statistics':
        showStat(total_df)
    elif selected == 'Map':
        showMap(total_df)
    else:
        st.warning("Wrong")