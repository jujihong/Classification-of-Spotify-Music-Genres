import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# 데이터 로드
Table = pd.read_csv("preprocessed.csv")

# 문자열을 범주형 데이터로 변환
Table['playlist_genre'] = LabelEncoder().fit_transform(Table['playlist_genre'])

# 특성 데이터 설정
data = Table.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

# 데이터 정규화
scaler = StandardScaler()
Table[data] = scaler.fit_transform(Table[data])

# 데이터 준비
X = Table[data]  # 특성 데이터
Y = Table['playlist_genre']  # 레이블 데이터

# 데이터를 훈련 세트와 테스트 세트로 분할
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.4, random_state=42)

# 데이터 균형 조정
sm = SMOTE(random_state=42)
XTrain, YTrain = sm.fit_resample(XTrain, YTrain)

# 특성 중요도 평가
Mdl = RandomForestClassifier(n_estimators=100, min_samples_split=2, max_features=0.1, random_state=42)
Mdl.fit(XTrain, YTrain)
importances = Mdl.feature_importances_
feature_importances = pd.Series(importances, index=X.columns)

# 중요도가 0.08 미만인 특성 제거
mask = feature_importances > 0.08
reduced_X = X.loc[:, mask]

# 제거된 특성 출력
removed_features = X.columns[~mask]
print("Removed features:")
print(removed_features)

# 데이터를 훈련 세트와 테스트 세트로 분할
XTrain, XTest, YTrain, YTest = train_test_split(reduced_X, Y, test_size=0.4, random_state=41)

# 데이터 균형 조정
sm = SMOTE(random_state=42)
XTrain, YTrain = sm.fit_resample(XTrain, YTrain)

#모델 훈련
Mdl = RandomForestClassifier(n_estimators=2000, min_samples_split=2, max_features=0.1, random_state=41)
Mdl.fit(XTrain, YTrain)

# 테스트 데이터에 대한 예측 수행
YPred = Mdl.predict(XTest)

# 정확도 계산
accuracy = accuracy_score(YTest, YPred)
print('Accuracy: %.2f%%' % (accuracy * 100))

# 혼동 행렬
print(confusion_matrix(YTest, YPred))
