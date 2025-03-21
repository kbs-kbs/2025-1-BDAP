import streamlit as st
import seaborn as sns
import pandas as pd

@st.cache_data
def load_data():
    df = sns.load_dataset('iris')
    return df

def main():
    st.title("Data Display - st.dataframe()")
    
    use_container_width = st.checkbox("Use container width", value=False)
    
    iris = load_data()
    
    st.dataframe(iris, use_container_width=use_container_width)
    
    st.dataframe(iris.iloc[:5, 0:3].style.highlight_max(axis=0))
    
if __name__ == "__main__":
    main()