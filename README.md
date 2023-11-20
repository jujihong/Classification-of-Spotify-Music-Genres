# AI+X 딥러닝
AI+X 딥러닝 Final Project



-----------------------------
### Title: Spotify 음원의 장르 분류
> 음원들의 여러가지 특성들을 이용해 음원의 장르 분류
### Members:
  - 주지홍 미래자동차공학과(tjqjtjqj@hanyang.ac.kr)
  - 심동빈 기계공학과(dong-bin@naver.com)
  - 유경원 경영학부(yoowon0527@naver.com)

----------------------------



## I. Proposal (Option 1)
### Motivation : 
  - 음악 장르 분류의 중요성: 음악은 다양한 장르와 스타일을 가지고 있으며, 이를 자동으로 분류할 수 있다면 이는 음악 추천 시스템, 음악 플랫폼의 개인화, 라디오 스테이션 구성 등 다양한 응용 분야에서 보다 나은 사용자 경험을 제공할 수 있습니다.
  - 데이터의 다양성: 다양한 음악 특징을 수집하고 활용하여, 음악을 듣는 사람들에게 최적으로 맞춰진 음악 추천을 제공할 수 있습니다.

### Goal:
  - 음악 장르 분류 모델 개발: 주어진 음악 특징 데이터를 활용하여 머신 러닝 모델을 개발하여 음악의 장르를 자동으로 분류합니다.
  - 모델의 정확성 향상: 특히 음악의 다양한 측면을 반영하는 다양한 특징을 고려하여 모델의 정확성을 향상시키고, 다양한 장르에 대한 분류 성능을 최적화합니다.

------------------------------------
## Datasets
  - non_preprocessed.csv: 전처리 이전의 데이터(https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs/data)
  - preprocessed.csv: 전처리 이후의 데이터
    
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
```matlab
% 테이블 생성
Table = readtable("preprocessed.csv");

>이는 같은 디렉토리 내의 preprocessed.csv 파일을 matlab 작업공간 내로 불러오는 코드입니다.

% 문자열을 범주형 데이터로 변환
Table.playlist_genre = categorical(Table.playlist_genre);

% 범주형 데이터를 숫자로 변환
Table.playlist_genre = double(Table.playlist_genre);
>이는 playlist_genre의 문자열 클래스를 숫자 클래스로 변경하는 전처리 과정입니다.

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
'''
