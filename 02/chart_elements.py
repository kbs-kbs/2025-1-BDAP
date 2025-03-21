import streamlit as st
import pandas as pd
import numpy as np

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["A", "B", "C"]
)

st.title("Streamlit 기본 차트")

st.subheader("Line chart")
st.line_chart(chart_data)

st.subheader("Area chart")
st.area_chart(chart_data)

st.subheader("Bar chart")
st.bar_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)

st.subheader("Map")
st.map(map_data) 
