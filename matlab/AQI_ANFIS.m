
data = readtable('..\data\processed\cleaned_aqi.csv');

% Select input features and target
feature_columns = {'PM2_5','PM10','NO2','CO','O3','AQI_lag1'};
inputs = data{:, feature_columns};
targets = data{:, 'AQI'};  % AQI as target


% Split Train(80%) and Test(20%) Set
n = size(inputs,1);
idx = floor(0.8 * n);

X_train = inputs(1:idx,:);
y_train = targets(1:idx);

X_test = inputs(idx+1:end,:);
y_test = targets(idx+1:end);


% Generate Initial FIS (Sugeno)
% 2 Membership Functions per input 
numMFs = 2;         
mfType = 'gaussmf';  

% Generate FIS structure automatically
fis = genfis1([X_train y_train], numMFs, mfType);

% Train ANFIS Model
numEpochs = 50;
anfis_model = anfis([X_train y_train], fis, [numEpochs 0 0.01 0.9 1.1]);


% Predict on test set
X_test = max(X_test, min(X_train));  
X_test = min(X_test, max(X_train));  

y_pred = evalfis(anfis_model, X_test);  


% Calculate RMSE and R^2
rmse = sqrt(mean((y_test - y_pred).^2));
r2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);

fprintf('ANFIS RMSE: %.2f\n', rmse);
fprintf('ANFIS R^2: %.3f\n', r2);


% Plot Predicted vs Actual

figure;
plot(y_test, 'b','LineWidth',1.5);
hold on;
plot(y_pred, 'r','LineWidth',1.5);
xlabel('Test Samples');
ylabel('AQI');
legend('Actual AQI','Predicted AQI');
title('ANFIS Prediction vs Actual AQI');
grid on;

% Plot membership functions for first input (PM2.5)
figure;
[x, mf] = plotmf(anfis_model, 'input', 1);
plot(x, mf, 'LineWidth',1.5);
xlabel('PM2.5');
ylabel('Membership Degree');
title('Input MF: PM2.5');
grid on;
