import streamlit as st
def main():
    
    st.title("Hello world")
    st.write("Hello world")
    x = st.slider('x')
    st.write(x, "x^2 =", x + 2)
    
if __name__ == "__main__":
    main()