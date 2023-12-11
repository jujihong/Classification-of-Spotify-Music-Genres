# AI+X 딥러닝
AI+X 딥러닝 Final Project



-----------------------------
### Title: Spotify 음원의 장르 분류
> 음원들의 여러가지 특성들을 이용해 음원의 장르 분류
### Members:
  - 주지홍 미래자동차공학과(tjqjtjqj@hanyang.ac.kr): python, matlab 코딩
  - 심동빈 기계공학과(dong-bin@naver.com): 자료조사, 제출 관련 작업 및 code review
  - 유경원 경영학부(yoowon0527@naver.com): 자료조사, github blog 관리 및 code review

----------------------------



## I. Proposal (Option 1)
### Motivation : 
  - 음악 장르 분류의 중요성: 음악은 다양한 장르와 스타일을 가지고 있으며, 이를 자동으로 분류할 수 있다면 이는 음악 추천 시스템, 음악 플랫폼의 개인화, 라디오 스테이션 구성 등 다양한 응용 분야에서 보다 나은 사용자 경험을 제공할 수 있습니다.
  - 데이터의 다양성: 다양한 음악 특징을 수집하고 활용하여, 음악을 듣는 사람들에게 최적으로 맞춰진 음악 추천을 제공할 수 있습니다.

### Goal:
  - 음악 장르 분류 모델 개발: 주어진 음악 특징 데이터를 활용하여 머신 러닝 모델을 개발하여 음악의 장르를 자동으로 분류합니다.
  - 모델의 정확성 향상: 특히 음악의 다양한 측면을 반영하는 다양한 특징을 고려하여 모델의 정확성을 향상시키고, 다양한 장르에 대한 분류 성능을 최적화합니다.

------------------------------------
## II. Datasets
  - non_preprocessed.csv: 전처리 이전의 데이터(https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs/data)
  - preprocessed.csv: 전처리 이후의 데이터(직접 엑셀 파일에서 누락 데이터 등을 제거하고, 추가로 matlab 또는 python을 통해 전처리를 진행)
    
      danceability는 음악의 템포, 리듬 안정성, 비트 강도 및 전반적인 규칙성을 기반으로 춤에 적합한 정도를 나타내는 지표입니다. 0.0은 가장 춤에 적합하지 않고, 1.0은 가장 춤에 적합한 것을 나타냅니다.

      energy는 0.0에서 1.0까지의 측정값으로, 강도와 활동성을 지각적으로 나타냅니다. 일반적으로 활기찬 트랙은 빠르고 시끄럽게 느껴집니다. 예를 들어, 데스 메탈은 높은 에너지를 가지고 있지만 바흐의 전주곡은 저 에너지 스케일에서 점수가 낮습니다.

      key는 트랙의 전반적인 음조를 추정한 값으로, 정수는 표준 음계 표기를 사용합니다. 예를 들어, 0은 C, 1은 C♯/D♭, 2는 D, 이런 식입니다. 만약 음조를 감지하지 못한 경우 값은 -1입니다.

      loudness는 트랙 전체의 데시벨(dB)로 나타낸 전체적인 음량입니다. 라우드니스 값은 전체 트랙에 걸쳐 평균화되며 트랙 간 상대적인 음량을 비교하는 데 유용합니다. 값은 일반적으로 -60에서 0 dB 사이에 있습니다.

      mode는 트랙의 조성(메이저 또는 마이너)을 나타내며, 멜로디 콘텐츠가 파생된 음계의 유형을 나타냅니다. 메이저는 1로 표시되고 마이너는 0입니다.

      speechiness는 트랙에서 말이 포함되어 있는 정도를 감지합니다. 녹음이 대화형(토크쇼, 오디오북, 시 등)일수록 속성 값은 1.0에 가까워집니다.

      acousticness는 트랙이 얼마나 아쿠스틱한지에 대한 0.0에서 1.0의 신뢰도 측정값입니다. 1.0은 트랙이 고도로 아쿠스틱하다는 것을 나타냅니다.

      instrumentalness는 트랙에 보컬이 포함되어 있는지 여부를 예측합니다. 값이 1.0에 가까울수록 보컬 콘텐츠가 없을 가능성이 큽니다.

      liveness는 녹음에서 관객의 존재를 감지합니다. 높은 liveness 값은 트랙이 라이브로 실행되었을 가능성이 크다는 것을 나타냅니다.

      valence는 트랙이 전하는 음악적 긍정성을 나타내는 0.0에서 1.0의 측정값입니다. 높은 valence는 더 긍정적인(행복한, 즐거운, 환상적인) 트랙을 나타내며, 낮은 valence는 더 부정적인(슬픈, 우울한, 화가 나는) 트랙을 나타냅니다.

      tempo는 분당 비트 수로 나타낸 전체적인 추정 템포입니다. 음악 용어에서 템포는 주어진 곡의 속도나 페이스를 직접적으로 나타냅니다.

      duration_ms는 곡의 재생 시간을 밀리초 단위로 나타낸 것입니다.

------------------------------------
## III. Methodology
### Preprocessing
  - 먼저 preprocessed.csv에서 학습에 관련이 없는 track_id,	track_name,	track_artist,	track_popularity,	track_album_id,	track_album_name,	track_album_release_date,	playlist_name,	playlist_id, playlist_subgenre는 제거하였습니다.
  - playlist_genre의 데이터 중 숫자로 이루어진 데이터(잘못된 데이터, 결측치)가 있으므로 이 값들을 제거하였습니다.
  - playlist_genre의 데이터 중 edm, pop, latin, r&b, rap, rock을 제외한 데이터는 앞의 여섯가지 데이터에 비해 데이터의 수가 현저히 적기에 클래스에서 제외하였습니다.
  - playlist_genre의 dem, pop, latin, r&b, rap, rock의 분류를 위해 1에서 6의 숫자 클래스를 적용하였습니다.

### Machine learning method
  - 결정 트리 모델
  - 랜덤 포레스트 모델

### Code 설명
#### 결정 트리 모델
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/main/Decision_tree.jpg">
</p>

결정 트리는 데이터를 기반으로 예측 모델을 만드는 알고리즘으로, 의사 결정 과정을 트리 구조로 나타냅니다. 뿌리 노드에서 시작하여 결정 노드를 통해 데이터를 분할하고, 잎 노드에서 예측 결과를 제공합니다. 학습 과정에서 중요한 속성을 찾아내어 이를 기준으로 데이터를 계층적으로 분류하는 방식으로 작동합니다.
```matlab
% 테이블 생성
Table = readtable("preprocessed.csv");
```
>같은 디렉토리 내의 preprocessed.csv 파일을 matlab 작업공간 내로 Table이라는 table 데이터 형식으로 불러오는 코드입니다.
```matlab
% 문자열을 범주형 데이터로 변환
Table.playlist_genre = categorical(Table.playlist_genre);

% 범주형 데이터를 숫자로 변환
Table.playlist_genre = double(Table.playlist_genre);
```
>playlist_genre의 문자열 클래스를 숫자 클래스로 변경하는 전처리 과정입니다.
```matlab
% 특성 데이터 설정
data = 2:13;

% 데이터 준비
X = Table(:, data); % 특성 데이터
Y = Table.playlist_genre; % 레이블 데이터

% 데이터를 훈련 세트와 테스트 세트로 분할
cv = cvpartition(size(Table, 1), 'HoldOut', 0.2);
idx = cv.test;

% 훈련과 테스트 데이터 분할
XTrain = X(~idx,:);
YTrain = Y(~idx,:);
XTest = X(idx,:);
YTest = Y(idx,:);
```
>table의 2에서 13열을 특성 데이터로 설정하고, 데이터를 훈련 세트와 테스트 세트로 분할하는 코드입니다. 전체 데이터의 80%가 train 데이터, 나머지 20% 데이터가 test 데이터로 이용됩니다.
```matlab
% 결정 트리 모델 훈련
Mdl = fitctree(XTrain,YTrain);

% 테스트 데이터에 대한 예측 수행
YPred = predict(Mdl, XTest);
```
>train용 특성 데이터와 레이블 데이터를 이용하여 결정 트리 모델을 훈련하고, test용 특성 데이터와 레이블 데이터와 결정 트리 모델을 통해 예측을 수행합니다.
```matlab
% 정확도 계산
accuracy = sum(YTest == YPred) / length(YTest);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
```
>정확도를 계산합니다. 정확도는 올바로 예측한 횟수/전체 예측 횟수*100(%)으로 표현됩니다.
```matlab
% 혼동 행렬
confusionchart(YTest, YPred);
```
>혼동 행렬을 그립니다. 혼동 행렬(Confusion Matrix)은 머신 러닝과 통계에서 분류 모델의 성능을 평가하는 데 사용되는 표입니다. 이 행렬은 모델이 예측한 결과와 실제 결과 간의 일치 및 불일치를 요약하여 제공합니다.
test 데이터와 train 데이터를 나누는 것은 랜덤하게 이루어짐으로, 시뮬레이션의 매 실행마다 정확도 결과는 달라집니다. 
한 번의 실행 결과에 대한 결과 파일은 decisiontree.mat입니다. 아래의 코드를 통해 같은 디렉토리 내의 mat 파일을 매트랩 작업 공간 내로 불러올 수 있습니다.
```matlab
% 파일 불러오기
load('decisiontree.mat');
```
>파일을 작업 공간으로 불러온 후 정확도 계산 및 혼동 행렬을 실행하면 정확도 45.1%와 아래의 혼동 행렬을 얻을 수 있습니다.
><p align="center">
  <img src="https://github.com/jujihong/predict_demagnet/blob/main/decisiontree_confusion_matrix.jpg">
</p>

#### 랜덤 포레스트 모델
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/main/Random_forest.jpg">
</p>
6개의 클래스가 있기에, test 데이터를 무작위로 찍으면 결과가 정답일 확률은 16.66%입니다. 결정 트리 모델은 대략 44~46% 정도의 정확도를 보였습니다. 이는 16.66%에 비해서는 높은 값이지만, 좀 더 좋은 모델을 이용하여 분류의 정확도를 높이고자 하였습니다.

랜덤 포레스트(Random Forest)는 여러 개의 결정 트리(Decision Tree)를 조합하여 높은 성능과 안정성을 제공하는 앙상블(Ensemble) 학습 모델입니다. 랜덤 포레스트는 다수의 결정 트리를 만들고 각 트리의 예측을 결합함으로써 높은 정확도를 달성합니다.

랜덤 포레스트 모델은 python을 통하여 코딩하였습니다.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
```
> 필요한 라이브러리를 import합니다.
pandas: 데이터 분석을 위한 파이썬 라이브러리로, 데이터프레임(DataFrame)이라는 효율적인 데이터 구조를 제공합니다. pd.read_csv 함수는 CSV 파일을 읽어 데이터프레임으로 변환하는 데 사용됩니다.

numpy: 수치 계산을 위한 파이썬 라이브러리로, 다차원 배열 객체와 이를 다루기 위한 도구를 제공합니다.

sklearn.ensemble.RandomForestClassifier: 랜덤 포레스트 분류 모델을 구현한 클래스입니다. 랜덤 포레스트는 여러 개의 결정 트리를 학습시키고, 그 결과를 평균내어 예측하는 알고리즘입니다.

sklearn.model_selection.train_test_split: 데이터를 훈련 세트와 테스트 세트로 분할하는 함수입니다.

sklearn.metrics.confusion_matrix: 혼동 행렬(confusion matrix)을 계산하는 함수입니다. 혼동 행렬은 분류 모델의 성능을 평가하는 데 사용됩니다.

sklearn.preprocessing.LabelEncoder: 범주형 레이블을 정수로 변환하는 데 사용되는 클래스입니다.

sklearn.preprocessing.StandardScaler: 특성의 스케일을 조정하는 데 사용되는 클래스입니다. 각 특성의 평균을 0, 분산을 1로 변경하여 정규화합니다.

sklearn.metrics.accuracy_score: 분류 모델의 정확도를 계산하는 함수입니다.

imblearn.over_sampling.SMOTE: SMOTE(Synthetic Minority Over-sampling Technique)는 소수 클래스의 오버샘플링을 위한 기법입니다. 소수 클래스의 샘플을 임의로 선택하고 이웃하는 샘플들 사이에 가상의 샘플을 생성합니다. 이를 통해 클래스 불균형 문제를 해결할 수 있습니다.

```python
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
```
>결정 트리 모델(matlab)에서의 과정과 동일합니다.
```python
# 데이터 균형 조정
sm = SMOTE(random_state=42)
XTrain, YTrain = sm.fit_resample(XTrain, YTrain)
```
>이 코드는 데이터의 클래스 불균형을 조정하기 위해 사용됩니다. 소수 클래스의 샘플을 임의로 선택하고 이웃하는 샘플들 사이에 가상의 샘플을 생성하는 오버샘플링 방법입니다.
```python
# 특성 중요도 평가
Mdl = RandomForestClassifier(n_estimators=300, min_samples_split=2, max_features=0.1, random_state=42)
Mdl.fit(XTrain, YTrain)
importances = Mdl.feature_importances_
feature_importances = pd.Series(importances, index=X.columns)
```
>랜덤 포레스트 분류기 객체를 생성합니다. 이때 n_estimators는 생성할 트리의 개수, min_samples_split는 노드를 분할하기 위한 최소의 샘플 수,max_features는 각 분할에서 고려할 특성의 최대 수, random_state는 결과의 재현성을 보장하기 위한 난수 생성기의 시드입니다. 그 후, 랜덤 포레스트 분류기 모델을 훈련 데이터를 통해 학습시킵니다. 다음으로 훈련된 분류기의 특성 중요도를 가져옵니다. 특성 데이터 12개중 필요 없는 데이터를 제외하기 위함입니다.

```python
# 중요도가 0.08 미만인 특성 제거
mask = feature_importances > 0.08
reduced_X = X.loc[:, mask]
```
>중요도가 0.08 미만인 특성을 제거합니다. 여러 번의 실행을 통해 중요도 경계를 조절하였습니다. 모델의 분류 정확도를 기준으로 경계의 값이 결정되었습니다.

```python
# 제거된 특성 출력
removed_features = X.columns[~mask]
print("Removed features:")
print(removed_features)
```
>prompt 창에서 제거된 특성을 확인할 수 있도록 합니다.

```python
# 데이터를 훈련 세트와 테스트 세트로 분할
XTrain, XTest, YTrain, YTest = train_test_split(reduced_X, Y, test_size=0.4, random_state=41)

# 데이터 균형 조정
sm = SMOTE(random_state=42)
XTrain, YTrain = sm.fit_resample(XTrain, YTrain)

#모델 훈련
Mdl = RandomForestClassifier(n_estimators=300, min_samples_split=3, max_features=0.1, random_state=41)
Mdl.fit(XTrain, YTrain)

# 테스트 데이터에 대한 예측 수행
YPred = Mdl.predict(XTest)
```
>X의 데이터가 변경되었으므로(삭제된 데이터가 존재하므로), 다시 데이터를 분할합니다. 분할할 때의 랜덤 시드는 이전의 분할과 다른 값으로 선택되었습니다(42>41). 이후 랜덤 포레스트 분류기 객체를 생성하고, 모델을 훈련하고, 훈련된 모델을 이용하여 예측을 진행합니다.
```python
# 정확도 계산
accuracy = accuracy_score(YTest, YPred)
print('Accuracy: %.2f%%' % (accuracy * 100))

# 혼동 행렬
print(confusion_matrix(YTest, YPred))
```
>정확도와 혼동 행렬을 표시합니다.
><p align="center">
  <img src="https://github.com/jujihong/predict_demagnet/blob/main/result_randomforest.jpg">
</p>
> 특성 데이터중 key, mode, liveness가 특성 중요도에 의해 삭제되었음을 확인할 수 있습니다.
또한 55.7%의 정확도 결과가 나왔습니다.

## IV. Conclusion
  - 결정 트리 모델을 사용하여 분류를 진행하였을 때보다, 결정 트리 모델의 앙상블 모델인 랜덤 포레스트 모델을 진행하였을 경우, 더 높은 정확도를 얻을 수 있었습니다. 이외에도 여러 랜덤 포레스트 모델을 이용하는 보팅 방법, 랜덤 포레스트 모델의 하이퍼 파라미터 튜닝 방법(GridSearchCV, RandomizedSearchCV), 신경망 방법 등을 이용하여 분류를 진행해 보았으나, 두 번째로 표현된 모델이 가장 높은 정확도를 가지는 것을 확인하였습니다.
  - 신경망 방법으로는(neural.m) 약 22% 정도의 정확도를 얻을 수 있었습니다. 이는 랜덤으로 분류 시의 정확도(16.6%)보다 크게 높은 값이 아니므로 좋은 모델이라고 할 수 없습니다.
  - 랜덤 포레스트 모델을 이용한 보팅 방법, 랜덤 포레스트 모델의 하이퍼 파라미터 튜닝 방법(GridSearchCV, RandomizedSearchCV)은 정확도가 약 53~54%로 나왔습니다.

  - 사용한 데이터가 12개의 특성 데이터를 가지고 있고, 전처리 이후에도 6개의 클래스를 가지는 매우 어려운 분류이므로, 높은 정확도를 얻을 수는 없었습니다. 같은 클래스에서 가지는 특성 데이터의 값의 이상치를 제거하려는 노력또한 해보았으나, 유의미한 결과를 얻을 수는 없었습니다. 더 많은 지식과, 데이터 셋에대한 깊은 이해가 좋은 모델을 만들기 위해서는 필요할 것이라 생각됩니다.
  - 랜덤 포레스트 모델에서 특성 중요도에 따라 특성 데이터가 삭제되었는데, 각 클래스 별 특성 데이터의 히스토그램을 통해 이를 설명하려 하였습니다. 결과는 아래의 그림과 같습니다.
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/acousticness.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/danceability.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/duration_s.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/energy.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/instrumentalness.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/key.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/liveness.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/loudness.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/mode.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/speechiness.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/tempo.jpg?raw=true">
</p>
><p align="center">
  <img src="https://github.com/jujihong/Classification-of-Spotify-Music-Genres/blob/jpg/valence.jpg?raw=true">
</p>
  - key, mode, liveness가 특성 중요도 평가를 통해 제거되었기에, 클래스에 따른 특성 데이터의 분포의 히스토그램에서 클래스끼리 구별되는 특성이 거의 없을 것이라고 예측하였으나, 제거되지 않은 특성 데이터들 중에도 클래스별로 눈에 띄는 차이를 보이지 않는 경우도 있었기에, 히스토그램을 분석하여 특성 중요도와 데이터 분포간의 관계를 직관적으로 파악하는 것은 실패하였습니다.

## V. Reference
  - https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs/data: 데이터셋
  - https://kr.mathworks.com/help/stats/decision-trees.html: 결정 트리 방법(MATLAB)
  - https://kr.mathworks.com/help/stats/regression-tree-ensembles.html: 랜덤 포레스트 방법(MATLAB)
