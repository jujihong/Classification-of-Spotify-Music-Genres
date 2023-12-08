clear; clc;

% 테이블 생성
Table = readtable("preprocessed.csv");

% 문자열을 범주형 데이터로 변환
Table.playlist_genre = categorical(Table.playlist_genre);

% 범주형 데이터를 숫자로 변환
Table.playlist_genre = double(Table.playlist_genre);

% 특성 데이터 추출
features = table2array(Table(:, 2:13));

% 특성 데이터 설정(몇 열)
k=12; %2~13으로 설정

% 2열 특성 데이터 추출
feature2 = features(:, k-1);
class1_data = feature2(Table.playlist_genre == 1);
class2_data = feature2(Table.playlist_genre == 2);
class3_data = feature2(Table.playlist_genre == 3);
class4_data = feature2(Table.playlist_genre == 4);
class5_data = feature2(Table.playlist_genre == 5);
class6_data = feature2(Table.playlist_genre == 6);

% 클래스별 데이터 배열 생성
class_data = {class1_data, class2_data, class3_data, class4_data, class5_data, class6_data};

% 클래스 이름 배열
class_names = {'pop', 'rap', 'rock', 'ratin', 'r&b', 'edm'};

figure;
columnName = Table.Properties.VariableNames{k};
for i = 1:6
    subplot(1, 6, i);
    histogram(class_data{i}, 'Orientation', 'horizontal', 'DisplayStyle', 'stairs', 'LineWidth', 1, 'BinWidth', 0.005, 'Normalization', 'count');
    title(class_names{i});
    xlabel('Frequency');
    ylabel('Value');
end
sgtitle(Table.Properties.VariableNames{k});
