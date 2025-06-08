import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 페이지 설정
st.set_page_config(
    page_title="차량 엔진 고장 예측 시스템",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border: none;
    }
    .css-1d391kg {
        background-color: #2D2D2D;
    }
    /* 전체 폰트 크기 증가 */
    .stMarkdown, .stText, .stSelectbox, .stSlider, .stCheckbox {
        font-size: 1.2rem;
        color: #FFFFFF;
    }
    /* 제목 폰트 크기 증가 */
    h1 {
        font-size: 2.5rem !important;
        color: #FF4B4B !important;
    }
    h2 {
        font-size: 2rem !important;
        color: #FF4B4B !important;
    }
    h3 {
        font-size: 1.8rem !important;
        color: #FF4B4B !important;
    }
    /* 메트릭 폰트 크기 증가 */
    .stMetric {
        font-size: 1.3rem;
    }
    /* 체크박스 레이블 폰트 크기 */
    .stCheckbox label {
        font-size: 1.1rem;
    }
    /* 슬라이더 스타일 */
    .stSlider {
        color: #FF4B4B;
    }
    /* 프로그레스 바 스타일 */
    .stProgress > div > div {
        background-color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset/vehicle.csv')
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

# 모델 로드 함수
@st.cache_resource
def load_model():
    try:
        df = load_data()
        
        # 특성 선택
        X_columns = ['Vehicle_Model', 'Mileage', 'Reported_Issues', 'Vehicle_Age', 
                    'Battery_Status', 'Tire_Condition', 'Brake_Condition', 'Maintenance_History']
        
        # 범주형 변수 인코딩
        le_dict = {}
        df_encoded = df.copy()
        categorical_columns = ['Vehicle_Model', 'Battery_Status', 'Tire_Condition', 
                             'Brake_Condition', 'Maintenance_History']
        
        # NaN 값을 'Unknown'으로 대체하고 각 범주형 변수별로 LabelEncoder 학습
        for col in categorical_columns:
            df_encoded[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            le.fit(df_encoded[col].astype(str))
            df_encoded[col] = le.transform(df_encoded[col].astype(str))
            le_dict[col] = le
        
        # 모델 학습
        X = df_encoded[X_columns]
        y = df_encoded['Need_Maintenance']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 교차 검증으로 모델 정확도 계산
        cv_scores = cross_val_score(model, X, y, cv=5)
        model_accuracy = cv_scores.mean()
        
        return model, X_columns, le_dict, df, categorical_columns, model_accuracy
    except Exception as e:
        st.warning(f"모델 로드 중 오류 발생: {str(e)}")
        return None, None, None, None, None, None

# 변수 정보 정의
variable_info = {
    # 기본 정보
    'Vehicle_Model': {'type': 'categorical', 'unit': ''},
    'Mileage': {'type': 'numeric', 'unit': 'km', 'step': 1000, 'min': 0, 'max': 500000},
    'Reported_Issues': {'type': 'numeric', 'unit': '건', 'step': 1, 'min': 0, 'max': 20},
    'Vehicle_Age': {'type': 'numeric', 'unit': '년', 'step': 1, 'min': 0, 'max': 50},
    'Battery_Status': {'type': 'categorical', 'unit': ''},
    'Tire_Condition': {'type': 'categorical', 'unit': ''},
    'Brake_Condition': {'type': 'categorical', 'unit': ''},
    'Maintenance_History': {'type': 'categorical', 'unit': ''}
}

# 데이터 로드
df = load_data()
if df is None:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# 사이드바
st.sidebar.title("메뉴")
page = st.sidebar.radio("페이지 선택", ["입력", "예측 결과"])

# 페이지 선택
if page == "입력":
    st.title("차량 정보 입력")
    
    # 입력 데이터를 저장할 딕셔너리 초기화
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = {}
    
    # 입력 UI 생성
    st.subheader("기본 정보")
    for col in ['Vehicle_Model', 'Mileage', 'Reported_Issues', 'Vehicle_Age', 
                'Battery_Status', 'Tire_Condition', 'Brake_Condition', 'Maintenance_History']:
        if col in variable_info:
            var_type = variable_info[col]['type']
            var_unit = variable_info[col].get('unit', '')
            
            if var_type == 'categorical':
                # 범주형 변수는 드롭다운으로 처리
                unique_values = df[col].dropna().unique()
                if len(unique_values) > 0:
                    default_value = unique_values[0]
                    label = f"{col} ({var_unit})" if var_unit else col
                    st.session_state['input_data'][col] = st.selectbox(
                        label,
                        options=unique_values,
                        index=0
                    )
            else:
                # 수치형 변수는 슬라이더로 처리
                min_val = int(variable_info[col].get('min', 0))
                max_val = int(variable_info[col].get('max', 100))
                step = int(variable_info[col].get('step', 1))
                default_value = int(df[col].mean())
                
                # 기본값이 범위를 벗어나면 중간값으로 설정
                if default_value < min_val or default_value > max_val:
                    default_value = (min_val + max_val) // 2
                
                label = f"{col} ({var_unit})" if var_unit else col
                st.session_state['input_data'][col] = st.slider(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_value,
                    step=step
                )
    
    # 구분선 추가
    st.markdown("---")
    
    # 예측 시작 버튼
    if st.button("예측 시작", type="primary", use_container_width=True):
        st.session_state['page'] = "예측 결과"
        st.rerun()

# 예측 결과 페이지
elif page == "예측 결과":
    st.title("예측 결과 및 분석")
    
    if 'input_data' not in st.session_state:
        st.warning("먼저 입력 페이지에서 차량 정보를 입력해주세요.")
        st.stop()
    
    model, X_columns, le_dict, df, categorical_columns, model_accuracy = load_model()
    if model is None:
        st.warning("모델을 불러올 수 없어 예측을 수행할 수 없습니다.")
        st.stop()
    
    # 입력 데이터 준비
    input_data = st.session_state['input_data']
    
    # 예측 수행
    try:
        # 입력 데이터를 모델 형식에 맞게 변환
        input_df = pd.DataFrame([input_data])
        
        # 범주형 변수 인코딩
        for col in categorical_columns:
            if col in input_df.columns:
                # 학습된 LabelEncoder 사용
                le = le_dict[col]
                # 새로운 값이면 'Unknown'으로 처리
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except:
                    input_df[col] = le.transform(['Unknown'])[0]
        
        # 예측 확률 계산
        failure_probability = model.predict_proba(input_df[X_columns])[0][1]
        
        # 결과 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "정비 필요 확률",
                f"{failure_probability:.1%}",
                delta=None
            )
        
        with col2:
            # 위험도 표시
            if failure_probability >= 0.7:
                st.error("⚠️ 높은 위험")
            elif failure_probability >= 0.4:
                st.warning("⚠️ 주의 필요")
            else:
                st.success("✅ 안전")
        
        with col3:
            # 모델 정확도 표시
            st.metric(
                "모델 정확도",
                f"{model_accuracy:.1%}"
            )
        
        # 시각화
        st.subheader("정비 필요 확률 분포")
        
        # 전체 데이터의 정비 필요 확률 분포
        fig = px.histogram(
            df,
            x='Need_Maintenance',
            nbins=50,
            title='전체 차량의 정비 필요 확률 분포',
            color_discrete_sequence=['#FF4B4B']
        )
        fig.add_vline(
            x=failure_probability,
            line_dash="dash",
            line_color="yellow",
            annotation_text="현재 차량",
            annotation_position="top right"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 주요 변수별 정비 필요 확률 분석
        st.subheader("주요 변수별 정비 필요 확률 분석")
        
        # 연령대별 정비 필요 확률
        df['Age_Group'] = pd.qcut(df['Vehicle_Age'], q=5, labels=['매우 젊음', '젊음', '중간', '오래됨', '매우 오래됨'])
        age_maintenance = df.groupby('Age_Group')['Need_Maintenance'].mean().reset_index()
        
        fig_age = px.bar(
            age_maintenance,
            x='Age_Group',
            y='Need_Maintenance',
            title='차량 연령대별 평균 정비 필요 확률',
            color='Need_Maintenance',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # 주행거리별 정비 필요 확률
        df['Mileage_Group'] = pd.qcut(df['Mileage'], q=5, labels=['매우 적음', '적음', '중간', '많음', '매우 많음'])
        mileage_maintenance = df.groupby('Mileage_Group')['Need_Maintenance'].mean().reset_index()
        
        fig_mileage = px.bar(
            mileage_maintenance,
            x='Mileage_Group',
            y='Need_Maintenance',
            title='주행거리별 평균 정비 필요 확률',
            color='Need_Maintenance',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_mileage, use_container_width=True)
        
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
