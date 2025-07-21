import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 0. 웹 페이지 설정 (가장 먼저 실행되어야 합니다) ---
st.set_page_config(
    page_title="물 음용 가능성 예측 앱",
    page_icon="🔬",
    layout="centered", # wide 또는 centered
    initial_sidebar_state="auto"
)

# --- 번역 딕셔너리 정의 ---
# 모든 UI 텍스트를 담고 있는 딕셔너리입니다.
translations = {
    "ko": {
        "app_title": "물 음용 가능성 예측기",
        "app_description": "이 앱은 물의 다양한 성분 데이터를 기반으로 해당 물이 **마실 수 있는지 없는지**를 예측합니다. 아래 각 항목에 대한 값을 입력해주세요.",
        "input_section_title": "물 성분 입력",
        "ph": "pH (0 ~ 14)",
        "hardness": "Hardness (경도, 50 ~ 400 ppm)",
        "solids": "Solids (총 고형물, 5000 ~ 50000 ppm)",
        "chloramines": "Chloramines (염소아민, 1 ~ 10 ppm)",
        "sulfate": "Sulfate (황산염, 100 ~ 500 ppm)",
        "conductivity": "Conductivity (전도도, 100 ~ 800 μS/cm)",
        "organic_carbon": "Organic Carbon (유기 탄소, 2 ~ 20 ppm)",
        "trihalomethanes": "Trihalomethanes (THMs, 0 ~ 100 μg/L)",
        "turbidity": "Turbidity (탁도, 0 ~ 10 NTU)",
        "predict_button": "🚀 물 음용 가능성 예측하기",
        "prediction_result_title": "📊 예측 결과",
        "potable_message": "✅ **이 물은 '마실 수 있는 물(Potable)'입니다!**",
        "non_potable_message": "❌ **이 물은 '마실 수 없는 물(Non-potable)'입니다.**",
        "prediction_probability_title": "예측 확률:",
        "non_potable_proba": "마실 수 없음 (0):",
        "potable_proba": "마실 수 있음 (1):",
        "sidebar_title": "앱 정보",
        "sidebar_info": "이 앱은 Kaggle의 'Water Potability' 데이터셋을 기반으로 Random Forest 분류 모델을 사용하여 물의 음용 가능성을 예측합니다.",
        "developer_info": "개발자: 당신의 이름 또는 팀명",
        "file_not_found_error": "'water_potability.csv' 파일을 찾을 수 없습니다. 앱과 같은 디렉토리에 넣어주세요."
    },
    "en": {
        "app_title": "Water Potability Predictor",
        "app_description": "This app predicts whether water is **potable or not** based on various water quality parameters. Please enter the values for each item below.",
        "input_section_title": "Water Parameters Input",
        "ph": "pH (0 ~ 14)",
        "hardness": "Hardness (50 ~ 400 ppm)",
        "solids": "Solids (Total Dissolved Solids, 5000 ~ 50000 ppm)",
        "chloramines": "Chloramines (1 ~ 10 ppm)",
        "sulfate": "Sulfate (100 ~ 500 ppm)",
        "conductivity": "Conductivity (100 ~ 800 μS/cm)",
        "organic_carbon": "Organic Carbon (2 ~ 20 ppm)",
        "trihalomethanes": "Trihalomethanes (THMs, 0 ~ 100 μg/L)",
        "turbidity": "Turbidity (0 ~ 10 NTU)",
        "predict_button": "🚀 Predict Water Potability",
        "prediction_result_title": "📊 Prediction Result",
        "potable_message": "✅ **This water is 'Potable'!**",
        "non_potable_message": "❌ **This water is 'Non-potable'.**",
        "prediction_probability_title": "Prediction Probability:",
        "non_potable_proba": "Non-potable (0):",
        "potable_proba": "Potable (1):",
        "sidebar_title": "App Information",
        "sidebar_info": "This app predicts water potability using a Random Forest classification model based on the Kaggle 'Water Potability' dataset.",
        "developer_info": "Developer: Your Name or Team Name",
        "file_not_found_error": "'water_potability.csv' file not found. Please place it in the same directory as the app."
    }
}

# --- 1. 데이터 로딩 및 모델 학습 (한 번만 실행되도록 캐싱) ---
@st.cache_data
def load_data_and_train_model():
    try:
        df = pd.read_csv('water_potability.csv')
    except FileNotFoundError:
        # 파일이 없을 경우, Streamlit 에러 메시지를 표시하고 앱 실행 중단
        st.error(translations[st.session_state.lang]["file_not_found_error"])
        st.stop()

    # 결측치 처리
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 특성과 타겟 분리
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Random Forest 모델 학습
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)

    return model, X.columns

# --- 언어 선택 및 세션 상태 관리 ---
# 세션 상태에 'lang' 키가 없으면 기본값으로 'ko' (한국어) 설정
if 'lang' not in st.session_state:
    st.session_state.lang = 'ko'

# 사이드바에 언어 선택 드롭다운 추가
st.sidebar.subheader("Language / 언어")
selected_lang = st.sidebar.selectbox(
    "Select Language / 언어를 선택하세요",
    options=["ko", "en"],
    format_func=lambda x: "한국어" if x == "ko" else "English",
    key="lang_selector"
)

# 선택된 언어가 변경되면 세션 상태 업데이트
if selected_lang != st.session_state.lang:
    st.session_state.lang = selected_lang
    # 언어 변경 시 앱을 다시 로드하여 UI 텍스트를 업데이트
    st.rerun() # <-- 이 부분을 st.experimental_rerun()에서 st.rerun()으로 변경했습니다.

# 현재 선택된 언어에 맞는 텍스트 가져오기
_ = translations[st.session_state.lang]

# 모델과 특성 이름 로드/학습
# 언어가 변경되면 이 함수도 다시 실행되어야 하지만, st.cache_data 덕분에 파일 로딩/모델 학습은 한 번만 일어남
model, feature_columns = load_data_and_train_model()

# --- 2. 웹 앱 인터페이스 구성 ---

st.title(_["app_title"])
st.markdown("---")
st.write(_["app_description"])

st.markdown(f"### {_['input_section_title']}")

# 각 입력 필드에 대한 설명과 함께 입력 위젯 배치
col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input(_["ph"], min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input(_["hardness"], min_value=50.0, max_value=400.0, value=200.0, step=0.1)
    solids = st.number_input(_["solids"], min_value=5000.0, max_value=50000.0, value=25000.0, step=100.0)

with col2:
    chloramines = st.number_input(_["chloramines"], min_value=1.0, max_value=10.0, value=5.0, step=0.01)
    sulfate = st.number_input(_["sulfate"], min_value=100.0, max_value=500.0, value=250.0, step=0.1)
    conductivity = st.number_input(_["conductivity"], min_value=100.0, max_value=800.0, value=400.0, step=0.1)

with col3:
    organic_carbon = st.number_input(_["organic_carbon"], min_value=2.0, max_value=20.0, value=10.0, step=0.01)
    trihalomethanes = st.number_input(_["trihalomethanes"], min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    turbidity = st.number_input(_["turbidity"], min_value=0.0, max_value=10.0, value=5.0, step=0.01)

st.markdown("---")

# 예측 버튼
if st.button(_["predict_button"]):
    # 사용자 입력값을 DataFrame 형태로 변환
    user_data = pd.DataFrame([[
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]], columns=feature_columns)

    # 모델을 사용하여 예측
    prediction = model.predict(user_data)
    proba = model.predict_proba(user_data)

    st.markdown(f"### {_['prediction_result_title']}")
    if prediction[0] == 1:
        st.success(_["potable_message"])
        st.balloons()
    else:
        st.error(_["non_potable_message"])

    st.write("---")
    st.subheader(_["prediction_probability_title"])
    st.info(f"{_['non_potable_proba']} **{proba[0][0]*100:.2f}%**")
    st.info(f"{_['potable_proba']} **{proba[0][1]*100:.2f}%**")

st.markdown("---")
st.sidebar.title(_["sidebar_title"])
st.sidebar.info(_["sidebar_info"])
st.sidebar.markdown("---")
st.sidebar.text(_["developer_info"])
