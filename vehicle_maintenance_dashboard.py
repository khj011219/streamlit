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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
import os
from fpdf import FPDF
import io

# .env 파일에서 이메일 설정 로드
load_dotenv()
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

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

# PDF 생성 함수
def create_prediction_pdf(prediction_result, vehicle_info, failure_probability, model_accuracy, input_data, df, X_columns, model):
    pdf = FPDF()
    pdf.add_page()
    
    # 기본 폰트 설정
    pdf.set_font('Arial', '', 12)
    
    # 제목
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 20, 'Vehicle Maintenance Prediction Result', ln=True, align='C')
    pdf.ln(10)
    
    # 차량 정보
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Vehicle Information', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Vehicle Model: {vehicle_info["Vehicle_Model"]}', ln=True)
    pdf.cell(0, 10, f'Mileage: {vehicle_info["Mileage"]:,} km', ln=True)
    pdf.cell(0, 10, f'Vehicle Age: {vehicle_info["Vehicle_Age"]} years', ln=True)
    pdf.cell(0, 10, f'Reported Issues: {vehicle_info["Reported_Issues"]} cases', ln=True)
    pdf.ln(10)
    
    # 예측 결과
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Prediction Result', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Maintenance Probability: {failure_probability*100:.1f}%', ln=True)
    pdf.cell(0, 10, f'Model Accuracy: {model_accuracy*100:.1f}%', ln=True)
    pdf.ln(10)
    
    # 차량 상태 정보
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Vehicle Component Status', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Battery Status: {vehicle_info["Battery_Status"]}', ln=True)
    pdf.cell(0, 10, f'Tire Condition: {vehicle_info["Tire_Condition"]}', ln=True)
    pdf.cell(0, 10, f'Brake Condition: {vehicle_info["Brake_Condition"]}', ln=True)
    pdf.cell(0, 10, f'Maintenance History: {vehicle_info["Maintenance_History"]}', ln=True)
    pdf.ln(10)
    
    # 주요 영향 요인
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Top 5 Influential Factors', ln=True)
    pdf.set_font('Arial', '', 12)
    
    # 특성 중요도 계산
    base_importances = model.feature_importances_
    numeric_cols = ['Mileage', 'Reported_Issues', 'Vehicle_Age']
    relative_values = np.ones(len(X_columns))
    
    for i, col in enumerate(X_columns):
        if col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                relative_values[i] = abs((input_data[col] - mean_val) / std_val)
    
    relative_importances = base_importances * relative_values
    
    importance_df = pd.DataFrame({
        'feature': X_columns,
        'importance': relative_importances,
        'input_value': [input_data[f] for f in X_columns]
    })
    
    top5 = importance_df.nlargest(5, 'importance')
    
    for _, row in top5.iterrows():
        pdf.cell(0, 10, f'{row["feature"]}: {row["importance"]*100:.1f}% (Input: {row["input_value"]})', ln=True)
    
    # PDF를 바이트로 변환
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        return pdf_bytes.encode('latin1')
    return pdf_bytes

# 이메일 전송 함수
def send_prediction_email(to_email, prediction_result, vehicle_info, failure_probability, model_accuracy, input_data, df, X_columns, model):
    # PDF 생성
    pdf_bytes = create_prediction_pdf(prediction_result, vehicle_info, failure_probability, model_accuracy, input_data, df, X_columns, model)
    
    # 이메일 메시지 생성
    msg = MIMEMultipart()
    msg['Subject'] = 'Vehicle Maintenance Prediction Result'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    
    # 이메일 본문
    body = """
    Hello,
    
    Please find attached the vehicle maintenance prediction result.
    
    Thank you.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    # PDF 첨부
    pdf_attachment = MIMEApplication(pdf_bytes, _subtype='pdf')
    pdf_attachment.add_header('Content-Disposition', 'attachment', filename='prediction_result.pdf')
    msg.attach(pdf_attachment)
    
    # 이메일 전송
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Email sending failed: {str(e)}")
        return False

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
    /* 전체 폰트 크기 증가 */
    .stMarkdown, .stText, .stSelectbox, .stSlider, .stCheckbox, .stTextInput, .stRadio, .stNumberInput {
        font-size: 1.2rem !important;
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
        font-size: 1.3rem !important;
    }
    /* 체크박스 레이블 폰트 크기 */
    .stCheckbox label {
        font-size: 1.1rem !important;
    }
    /* 슬라이더 스타일 */
    .stSlider {
        color: #FF4B4B;
    }
    /* 프로그레스 바 스타일 */
    .stProgress > div > div {
        background-color: #FF4B4B;
    }
    /* 버튼 스타일 */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        font-size: 1.2rem !important;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        height: auto !important;
        min-height: 40px !important;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border: none;
    }
    /* 입력 필드 스타일 */
    .stTextInput input {
        font-size: 1.2rem !important;
        padding: 10px !important;
        height: auto !important;
        min-height: 30px !important;
    }
    /* 라디오 버튼 스타일 */
    .stRadio label {
        font-size: 1.2rem !important;
        padding: 5px 0 !important;
    }
    .stRadio [data-baseweb="radio"] {
        transform: scale(1);
        margin: 5px 0 !important;
    }
    /* 체크박스 스타일 */
    .stCheckbox [data-baseweb="checkbox"] {
        transform: scale(1);
        margin: 5px 0 !important;
    }
    /* 셀렉트박스 스타일 */
    .stSelectbox [data-baseweb="select"] {
        font-size: 1.2rem !important;
        padding: 10px !important;
        height: auto !important;
        min-height: 30px !important;
    }
    /* 슬라이더 스타일 */
    .stSlider [data-baseweb="slider"] {
        padding: 10px 0 !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        transform: scale(1);
    }
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #2D2D2D;
        font-size: 1.2rem !important;
    }
    .sidebar .sidebar-content {
        font-size: 1.2rem !important;
    }
    /* 사이드바 라디오 버튼 */
    .sidebar .stRadio label {
        font-size: 1.2rem !important;
        padding: 5px 0 !important;
    }
    .sidebar .stRadio [data-baseweb="radio"] {
        transform: scale(1);
        margin: 5px 0 !important;
    }
    /* 사이드바 제목 */
    .sidebar h1, .sidebar h2, .sidebar h3 {
        font-size: 1.5rem !important;
        color: #FF4B4B !important;
    }
    /* 모든 텍스트 요소 */
    p, div, span, label {
        font-size: 1.2rem !important;
    }
    /* 메트릭 값 */
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
    }
    /* 경고 메시지 */
    .stAlert {
        font-size: 1.2rem !important;
        padding: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 데이터 로드
df = load_data()
if df is None:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# 모델 미리 로딩
model, X_columns, le_dict, df, categorical_columns, model_accuracy = load_model()
if model is None:
    st.warning("모델을 불러올 수 없어 예측을 수행할 수 없습니다.")
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

# 예측 결과 페이지
elif page == "예측 결과":
    st.title("예측 결과 및 분석")
    
    if 'input_data' not in st.session_state:
        st.warning("먼저 입력 페이지에서 차량 정보를 입력해주세요.")
        st.stop()
    
    # 입력 데이터 준비
    input_data = st.session_state['input_data']
    
    try:
        # 입력 데이터를 모델 형식에 맞게 변환
        input_df = pd.DataFrame([input_data])
        
        # 범주형 변수 인코딩
        for col in categorical_columns:
            if col in input_df.columns:
                le = le_dict[col]
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except:
                    input_df[col] = le.transform(['Unknown'])[0]
        
        # 예측 확률 계산
        failure_probability = model.predict_proba(input_df[X_columns])[0][1]
        
        # 결과 표시
        st.subheader("예측 결과")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "정비 필요 확률",
                f"{failure_probability*100:.1f}%",
                delta=None
            )
        
        with col2:
            if failure_probability >= 0.7:
                st.error("⚠️ 높은 위험")
            elif failure_probability >= 0.4:
                st.warning("⚠️ 주의 필요")
            else:
                st.success("✅ 안전")
        
        with col3:
            st.metric(
                "모델 정확도",
                f"{model_accuracy*100:.1f}%"
            )
        
        # 위험도 게이지 차트
        st.subheader("위험도 시각화")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = failure_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF4B4B"},
                'steps': [
                    {'range': [0, 40], 'color': "#00CC96"},
                    {'range': [40, 70], 'color': "#FFA15A"},
                    {'range': [70, 100], 'color': "#EF553B"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            },
            title = {'text': "정비 필요 위험도 (%)"}
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 차량 정보 표시
        st.subheader("입력된 차량 정보")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"차량 모델: {input_data['Vehicle_Model']}")
            st.write(f"주행거리: {input_data['Mileage']:,}km")
            st.write(f"차량 연식: {input_data['Vehicle_Age']}년")
            st.write(f"보고된 문제: {input_data['Reported_Issues']}건")
        
        # 차량 상태 시각화
        st.subheader("차량 상태 시각화")
        
        # 상태 데이터 준비
        components = ['배터리 상태', '타이어 상태', '브레이크 상태', '정비 이력']
        statuses = [
            str(input_data['Battery_Status']),
            str(input_data['Tire_Condition']),
            str(input_data['Brake_Condition']),
            str(input_data['Maintenance_History'])
        ]
        
        # 각 상태에 대한 색상 결정
        colors = []
        for i, status in enumerate(statuses):
            if i == 0:  # Battery_Status
                if status == 'Weak':
                    colors.append('#EF553B')
                elif status == 'New':
                    colors.append('#636EFA')
                elif status == 'Good':
                    colors.append('#00CC96')
                else:
                    colors.append('#636EFA')
            elif i in [1, 2]:  # Tire_Condition & Brake_Condition
                if status == 'New':
                    colors.append('#636EFA')
                elif status == 'Good':
                    colors.append('#00CC96')
                elif status == 'Worn Out':
                    colors.append('#EF553B')
                else:
                    colors.append('#636EFA')
            else:  # Maintenance_History
                if status == 'Good':
                    colors.append('#636EFA')
                elif status == 'Average':
                    colors.append('#00CC96')
                elif status == 'Poor':
                    colors.append('#EF553B')
                else:
                    colors.append('#636EFA')
        
        # 상태 시각화 그래프
        fig = go.Figure(data=[
            go.Bar(
                x=components,
                y=[1, 1, 1, 1],
                marker_color=colors,
                text=statuses,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='차량 부품별 상태',
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 모델 특성 중요도 시각화
        st.subheader("모델이 중요하게 본 요인")
        
        # 입력값에 따른 상대적 중요도 계산
        base_importances = model.feature_importances_
        
        # 수치형 변수만 정규화
        numeric_cols = ['Mileage', 'Reported_Issues', 'Vehicle_Age']
        relative_values = np.ones(len(X_columns))
        
        for i, col in enumerate(X_columns):
            if col in numeric_cols:
                # 수치형 변수는 정규화된 차이 계산
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:  # 0으로 나누기 방지
                    relative_values[i] = abs((input_data[col] - mean_val) / std_val)
        
        # 상대적 중요도 계산
        relative_importances = base_importances * relative_values
        
        # 중요도 데이터프레임 생성
        importance_df = pd.DataFrame({
            'feature': X_columns,
            'importance': relative_importances,
            'input_value': [input_data[f] for f in X_columns]
        })
        
        # 중요도 상위 5개만 추출
        top5 = importance_df.nlargest(5, 'importance')
        
        # bar chart로 시각화
        fig_imp = go.Figure(data=[
            go.Bar(
                x=top5['importance'],
                y=[f"{row['feature']}\n({row['input_value']})" for _, row in top5.iterrows()],
                orientation='h',
                marker_color='#FF4B4B',
                text=[f"{v*100:.1f}%" for v in top5['importance']],
                textposition='auto',
            )
        ])
        
        fig_imp.update_layout(
            title='예측에 영향을 준 주요 요인 (상위 5개)',
            xaxis_title='상대적 중요도',
            yaxis_title='특성 (입력값)',
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # 권장 조치사항 표시
        st.subheader("권장 조치사항")
        if failure_probability >= 0.7:
            st.error("""
            ⚠️ 즉시 정비가 필요합니다.
            - 가까운 정비소에서 상세 점검을 받으시기 바랍니다.
            - 안전을 위해 운행을 자제하시기 바랍니다.
            """)
        elif failure_probability >= 0.4:
            st.warning("""
            ⚠️ 정비가 권장됩니다.
            - 가까운 시일 내에 정비소 방문을 권장드립니다.
            - 정기적인 점검을 통해 상태를 모니터링하시기 바랍니다.
            """)
        else:
            st.success("""
            ✅ 정상 상태입니다.
            - 정기적인 점검을 통해 현재 상태를 유지하시기 바랍니다.
            - 평소와 다른 이상 징후가 발견되면 즉시 점검을 받으시기 바랍니다.
            """)
        
        # 구분선 추가
        st.markdown("---")
        
        # 이메일로 결과 받기 섹션
        st.subheader("결과 이메일로 받기")
        st.write("예측 결과를 이메일로 받아보시겠습니까?")
        email = st.text_input("이메일 주소를 입력하세요")
        
        if st.button("결과 이메일로 받기"):
            if email:
                prediction_result = {
                    'probability': failure_probability,
                    'risk_level': "High Risk" if failure_probability >= 0.7 else "Caution Required" if failure_probability >= 0.4 else "Safe",
                    'recommendation': "Immediate maintenance required" if failure_probability >= 0.7 else "Maintenance recommended" if failure_probability >= 0.4 else "Normal condition"
                }
                
                if send_prediction_email(email, prediction_result, input_data, failure_probability, model_accuracy, input_data, df, X_columns, model):
                    st.success("예측 결과가 이메일로 전송되었습니다!")
                else:
                    st.error("이메일 전송에 실패했습니다. 다시 시도해주세요.")
            else:
                st.warning("이메일 주소를 입력해주세요.")
        
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
