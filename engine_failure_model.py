import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. CSV 파일 불러오기 (Colab에서는 직접 업로드한 파일 이름 사용)
df = pd.read_csv('dataset/vehicle.csv')

# 2. 확장된 사용자 친화 입력 변수 목록
selected_columns = [
    'Vehicle_Model',         # 차량 유형
    'Mileage',               # 마일리지
    'Reported_Issues',       # 보고된 문제 수
    'Vehicle_Age',           # 차량 연식
    'Battery_Status',        # 배터리 상태
    'Tire_Condition',        # 타이어 상태
    'Brake_Condition',       # 브레이크 상태
    'Maintenance_History'    # 유지보수 이력
]

X = df[selected_columns].copy()
y = df['Need_Maintenance']

# 3. 범주형 변수 라벨 인코딩
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 4. 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 예측 및 성능 평가 출력
y_pred = model.predict(X_test)
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
print("📊 모델 성능 보고서:")
print(report_df)

# 7. 혼동 행렬 시각화
plt.figure(figsize=(6, 4))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix - Random Forest (Expanded Variables)')
plt.tight_layout()
plt.show()
