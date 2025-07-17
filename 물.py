import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# matplotlib, seaborn은 직접적인 웹 UI를 만들지는 않으므로 여기서는 제외 (특성 중요도 시각화는 웹 앱 외부에서 진행)
# from sklearn.model_selection import train_test_split # 모델 학습은 웹 앱 실행 전에 완료
# from sklearn.metrics import classification_report, confusion_matrix # 평가는 웹 앱 실행 전에 완료

# --- 0. 웹 페이지 설정 (가장 먼저 실행되어야 합니다) ---
st.set_page_config(
    page_title="물 음용 가능성 예측 앱",
    page_icon="🔬",
    layout="centered", # wide 또는 centered
    initial_sidebar_state="auto"
)

# --- 1. 데이터 로딩 및 모델 학습 (한 번만 실행되도록 캐싱) ---
# @st.cache_data 데코레이터를 사용하여 데이터를 불러오고 모델을 학습하는 과정을 캐싱합니다.
# 이렇게 하면 앱을 업데이트하거나 다시 로드할 때마다 이 부분이 다시 실행되지 않아 빠릅니다.
@st.cache_data
def load_data_and_train_model():
    # 파일 업로드 (Streamlit에서는 st.file_uploader 사용)
    # 실제 앱 배포 시에는 파일을 직접 앱에 포함하거나 클라우드 저장소에서 불러오는 것이 일반적입니다.
    # 여기서는 예시를 위해 파일을 앱과 같은 디렉토리에 있다고 가정합니다.
    try:
        df = pd.read_csv('water_potability.csv')
    except FileNotFoundError:
        st.error("'water_potability.csv' 파일을 찾을 수 없습니다. 앱과 같은 디렉토리에 넣어주세요.")
        st.stop() # 파일이 없으면 앱 실행 중단

    # 결측치 처리 (원본 코드와 동일)
    df.fillna(df.mean(numeric_only=True), inplace=True) # numeric_only=True 추가하여 경고 방지

    # 특성과 타겟 분리 (원본 코드와 동일)
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # 학습/테스트 데이터 분리 (모델 학습에만 필요, 예측 시에는 전체 모델 사용)
    # 여기서는 간단히 전체 데이터로 모델을 다시 학습시킵니다.
    # 실제로는 train_test_split을 유지하고 학습된 모델을 저장/로드하는 것이 좋습니다.
    
    # Random Forest 모델 학습 (원본 코드와 동일)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y) # 전체 데이터로 학습

    return model, X.columns # 모델과 특성(컬럼) 이름을 함께 반환

# 모델과 특성 이름 로드/학습
model, feature_columns = load_data_and_train_model()

# --- 2. 웹 앱 인터페이스 구성 ---

st.title("물 음용 가능성 예측기")
st.markdown("---") # 구분선
st.write("""
이 앱은 물의 다양한 성분 데이터를 기반으로 해당 물이 **마실 수 있는지 없는지**를 예측합니다.
아래 각 항목에 대한 값을 입력해주세요.
""")

st.markdown("### 물 성분 입력")

# 각 입력 필드에 대한 설명과 함께 입력 위젯 배치
# st.columns를 사용하여 입력 필드를 깔끔하게 정렬합니다.
col1, col2, col3 = st.columns(3) # 3개의 열로 나누어 입력 필드를 배치

with col1:
    ph = st.number_input("pH (0 ~ 14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input("Hardness (경도, 50 ~ 400 ppm)", min_value=50.0, max_value=400.0, value=200.0, step=0.1)
    solids = st.number_input("Solids (총 고형물, 5000 ~ 50000 ppm)", min_value=5000.0, max_value=50000.0, value=25000.0, step=100.0)

with col2:
    chloramines = st.number_input("Chloramines (염소아민, 1 ~ 10 ppm)", min_value=1.0, max_value=10.0, value=5.0, step=0.01)
    sulfate = st.number_input("Sulfate (황산염, 100 ~ 500 ppm)", min_value=100.0, max_value=500.0, value=250.0, step=0.1)
    conductivity = st.number_input("Conductivity (전도도, 100 ~ 800 μS/cm)", min_value=100.0, max_value=800.0, value=400.0, step=0.1)

with col3:
    organic_carbon = st.number_input("Organic Carbon (유기 탄소, 2 ~ 20 ppm)", min_value=2.0, max_value=20.0, value=10.0, step=0.01)
    trihalomethanes = st.number_input("Trihalomethanes (THMs, 0 ~ 100 μg/L)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    turbidity = st.number_input("Turbidity (탁도, 0 ~ 10 NTU)", min_value=0.0, max_value=10.0, value=5.0, step=0.01)

st.markdown("---")

# 예측 버튼
if st.button("🚀 물 음용 가능성 예측하기"):
    # 사용자 입력값을 DataFrame 형태로 변환
    user_data = pd.DataFrame([[
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]], columns=feature_columns) # 학습 시 사용된 컬럼 순서와 이름 유지

    # 모델을 사용하여 예측
    prediction = model.predict(user_data)
    proba = model.predict_proba(user_data)

    st.markdown("### 📊 예측 결과")
    if prediction[0] == 1:
        st.success("✅ **이 물은 '마실 수 있는 물(Potable)'입니다!**")
        st.balloons() # 축하 풍선 효과!
    else:
        st.error("❌ **이 물은 '마실 수 없는 물(Non-potable)'입니다.**")

    st.write("---")
    st.subheader("예측 확률:")
    st.info(f"마실 수 없음 (0): **{proba[0][0]*100:.2f}%**")
    st.info(f"마실 수 있음 (1): **{proba[0][1]*100:.2f}%**")

st.markdown("---")
st.sidebar.title("앱 정보")
st.sidebar.info("""
이 앱은 Kaggle의 'Water Potability' 데이터셋을 기반으로
Random Forest 분류 모델을 사용하여 물의 음용 가능성을 예측합니다.
""")
st.sidebar.markdown("---")
st.sidebar.text("개발자: 당신의 이름 또는 팀명")