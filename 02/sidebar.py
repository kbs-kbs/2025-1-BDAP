import streamlit as st

def main():
    # 사이드바 설정
    st.sidebar.title("Sidebar Controls")
    
    # 슬라이터
    slider_value = st.sidebar.slider("Adjust Value", min_value=0, max_value=100, value=50)
    st.write(f'선택된 값: {slider_value}')
    
    # 라디오 버튼
    radio_value = st.sidebar.radio('choose an option', ['option1', 'option2', 'option3'])
    st.write(f'선택된 옵션: {radio_value}')
    
    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader('upload a file', type=['csv', 'txt'])
    if uploaded_file:
        st.write(f'업로드된 파일: {uploaded_file.name}')
    
    # 텍스트 입력
    text_value = st.sidebar.text_input('enter text', value='hello streamlit!')
    st.write(f'입력된 테스트: {text_value}')
    
if __name__ == '__main__':
    main()