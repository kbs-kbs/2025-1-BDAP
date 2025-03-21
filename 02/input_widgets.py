import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

iris = sns.load_dataset('iris')

# 데이터 불러오기
@st.cache_data
def load_data():
    tips = sns.load_dataset("tips")
    return tips

# 랜덤포레스트 모델 실행
@st.cache_resource
def run_model(data, max_depth, min_samples_leaf):
    y = data['tip']
    X = data[['total_bill', 'size']]
    
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.3, random_state=42)
    )

    param_dist = {
        'max_depth': list(range(max_depth[0], max_depth[1])),
        'min_samples_leaf': [min_samples_leaf]
    }

    rf = RandomForestRegressor()
    model = RandomizedSearchCV(estimator=rf,
                               param_distributions=param_dist,
                               n_iter=10,
                               cv=4,
                               random_state=101,
                               n_jobs=-1)
    model.fit(X_train, y_train)

    return model.best_estimator_, X_test, y_test

def prediction(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    return y_test_pred, test_mae, r2

def prediction_plot(X_test, y_test, y_test_pred, test_mae, r2):
    fig = go.Figure()
    
    # 실제값
    fig.add_trace(
        go.Scatter(x=X_test['total_bill'], y=y_test, mode='markers', name='Actual', marker=dict(color='red'))
    )
    
    # 예측값
    fig.add_trace(
        go.Scatter(x=X_test['total_bill'], y=y_test_pred, mode='markers', name='Predicted', marker=dict(color='green'))
    )
    
    fig.update_layout(
        title="tip prediction with randomforest regressor",
        xaxis_title='total bill 달러',
        yaxis_title='tip 달러',
        annotations=[go.layout.Annotation(
            x=40, y=1.5, text=f'test mae: {test_mae:.3f} | R2 Score: {r2:.3f}',
            showarrow=False
        )]
    )
    st.plotly_chart(fig)

def plot_matplotlib():
    st.title("Scatter Plot with MatplotLib")
    fig, ax, = plt.subplots()
    ax.scatter(iris["sepal_length"], iris["sepal_width"])
    st.pyplot(fig)

def plot_seaborn():
    st.title("Scatter Plot with Seaborn")
    fig, ax = plt.subplots()
    sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", ax=ax)
    st.pyplot(fig)
    
def plot_plotly():
    st.title("Scatter Plot with Plotly")
    fig = go.Figure(data=go.Scatter(x=iris["sepal_length"], y=iris["sepal_width"], mode="markers"))
    st.plotly_chart(fig)

def main():
    st.title("Check Box를 활용한 시각화 제어")

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    show_plot = st.checkbox("시각화 보여주기")

    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, color="blue")
        ax.set_title("Sine Wave")
        st.pyplot(fig)
        
    st.title("플로팅 라이브러리 선택")
    plot_type = st.radio("어떤 스타일의 산점도를 보고 싶은가요?", ("Matplotlib", "Seaborn", "Plotly"))
    
    if plot_type == "Matplotlib":
        plot_matplotlib()
    elif plot_type == "Seaborn":
        plot_seaborn()
    elif plot_type == "Plotly":
        plot_plotly()
        
    st.title("머신러닝 하이퍼파라미터 튜닝")
    
    max_depth = st.select_slider('최대 깊이 선택', options=list(range(2, 30)), value=(5, 10))
    min_samples_leaf = st.slider('최소 샘플 리프', min_value=2, max_value=20)
    
    tips = load_data()
    
    # 모델 학습
    model, X_test, y_test = (
        run_model(tips, max_depth, min_samples_leaf)
    )
    
    # 예측 수행
    y_test_pred, test_mae, r2, = prediction(model, X_test, y_test)
    
    # 시각화
    prediction_plot(X_test, y_test, y_test_pred, test_mae, r2)
    
    
if __name__ == "__main__":
    main()