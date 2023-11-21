clear; clc;

% 테이블 생성
Table = readtable("preprocessed.csv");

% 문자열을 범주형 데이터로 변환
Table.playlist_genre = categorical(Table.playlist_genre);

% 범주형 데이터를 정수 인덱스로 변환
[Table.playlist_genre, genreNames] = grp2idx(Table.playlist_genre);

% 특성 데이터 설정
data = [2 3 5 7 8 9 11 12 13];

% 데이터 정규화
for i = data
    Table.(Table.Properties.VariableNames{i}) = normalize(Table.(Table.Properties.VariableNames{i}));
end

% 데이터 준비
X = table2array(Table(:, data)); % 특성 데이터를 배열로 변환
Y = Table.playlist_genre; % 레이블 데이터

% 데이터를 훈련 세트와 테스트 세트로 분할
cv = cvpartition(size(Table, 1), 'HoldOut', 0.4);
idx = cv.test;

% 훈련과 테스트 데이터 분할
XTrain = X(~idx,:);
YTrain = Y(~idx,:);
XTest = X(idx,:);
YTest = Y(idx,:);

% 신경망 생성
net = patternnet(100);

% 신경망 훈련
net = train(net, XTrain', YTrain');

% 테스트 데이터에 대한 예측 수행
YPred = net(XTest');

% 정확도 계산
accuracy = sum(YTest == round(YPred')) / length(YTest);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

%% 혼동 행렬
confusionchart(YTest, round(YPred));
