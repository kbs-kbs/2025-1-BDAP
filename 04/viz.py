import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

def meanChart(total_df, CGG_NM):
    st.markdown('## 가구별 평균 가격 추세')
    filtered_df = total_df[total_df['CGG_NM'] == CGG_NM]
    filtered_df = filtered_df[filtered_df['CTRT_DAY'].between("2025-02-01", "2025-03-31")]
    
    result = filtered_df.groupby(['CTRT_DAY', 'BLDG_USG'])['THING_AMT'].mean().reset_index()
    
    fig = make_subplots(rows=2, cols=2,
                        shared_xaxes=True,
                        subplot_titles=['아파트', '단독다가구', '오피스텔', '연립다세대'],
                        horizontal_spacing=0.15)
    
    house_types = ['아파트', '단독다가구', '오피스텔', '연립다세대']
    row_col = [(1, 1), (1,2), (2, 1), (2,2)]
    
    for i, house_type in enumerate(house_types):
        df = result[result['BLDG_USG'] == house_type]
        fig.add_trace(px.line(df, x='CTRT_DAY', y='THING_AMT').data[0],
                      row=row_col[i][0], col=row_col[i][1])
        
    st.plotly_chart(fig)
    
def barChart(total_df):
    st.markdown('### 지역별 평균 가격 막대 그래프')
    
    month_selected = st.selectbox('월을 선택하세요', [2, 3])
    house_selected = st.selectbox('가구 유형을 선택하세요', total_df['BLDG_USG'].unique())
    
    total_df['month'] = pd.to_datetime(total_df['CTRT_DAY']).dt.month
    result = total_df[(total_df['month'] == month_selected) & (total_df['BLDG_USG'] == house_selected)]
    
    bar_df = result.groupby('CGG_NM')['THING_AMT'].mean().reset_index()
    df_sorted = bar_df.sort_values('THING_AMT', ascending=False)