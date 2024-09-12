from matplotlib.pylab import f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import os
import contextlib
import pickle 
import tsfel

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from keras.src.layers import MaxPooling1D, Conv1D

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Redirect stderr to /dev/null to silence warnings
devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)

def create_sequences(X, Y, timesteps, unique_activities):
    Xin, Yin = [], []
    for activity in unique_activities:
        for i in range(0, len(X) - timesteps, timesteps // 2):
            if Y.iloc[i] != activity or Y.iloc[i + timesteps] != activity:
                continue

            Xin.append(X.iloc[i:(i + timesteps)].values)
            Yin.append(activity)
    Xin, Yin = np.array(Xin), np.array(Yin)
    return Xin, Yin.reshape(-1, 1)


def split_data(path, timesteps):
    data = pd.read_csv(path)
    data = data.drop(['timestamp'], axis=1)
    data = data.drop(['user_id'], axis=1)

    # Define a mapping of letters to numbers
    letter_to_number = {'BRUSHING_TEETH': 1, 'CYCLING': 2, 'ELEVATOR_DOWN': 3, 'ELEVATOR_UP': 4, 'LYING': 5,
                        'MOVING_BY_CAR': 6, 'RUNNING': 7, 'SITTING': 8, 'SITTING_ON_TRANSPORT': 9, 'STAIRS_DOWN': 10,
                        'STAIRS_UP': 11, 'STANDING': 12, 'STANDING_ON_TRANSPORT': 13, 'WALKING': 14, 'TRANSITION': 15}

    data['activityId'] = data['activity'].map(letter_to_number)

    # remove some activities
    undesired_activities = [3, 4, 9, 11, 13, 15]
    data = data[~data['activityId'].isin(undesired_activities)]

    unique_activities = data['activityId'].unique()

    # Transform the frequency of the data from 100Hz to 25Hz
    data = data.iloc[::4, :]

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    size = len(data)
    train_data = data.iloc[0:int(size*0.8)]
    test_data = data.iloc[int(size*0.8):]

    X_train, y_train = create_sequences(train_data[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']],
                                     train_data['activityId'], timesteps, unique_activities)
    X_test, y_test = create_sequences(test_data[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']], test_data['activityId'],
                                   timesteps, unique_activities)

    np.random.seed(42)

    random = np.arange(0, len(y_train))
    np.random.shuffle(random)
    X_train = X_train[random]
    y_train = y_train[random]

    random = np.arange(0, len(y_test))
    np.random.shuffle(random)
    X_test = X_test[random]
    y_test = y_test[random]

    # for activity in unique_activities:
    #     print(f'Activity {activity}: {len(y_train[y_train == activity])}')

    hotenc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    hotenc = hotenc.fit(y_train)
    y_train = hotenc.transform(y_train)
    y_test = hotenc.transform(y_test)

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


def extract_features(path, frequency, samples_required, train_features):
    data = pd.read_csv(path)
    data = data.drop(['timestamp'], axis=1)
    data = data.drop(['user_id'], axis=1)

    # Define a mapping of letters to numbers
    letter_to_number = {'BRUSHING_TEETH': 1, 'CYCLING': 2, 'ELEVATOR_DOWN': 3, 'ELEVATOR_UP': 4, 'LYING': 5,
                        'MOVING_BY_CAR': 6, 'RUNNING': 7, 'SITTING': 8, 'SITTING_ON_TRANSPORT': 9, 'STAIRS_DOWN': 10,
                        'STAIRS_UP': 11, 'STANDING': 12, 'STANDING_ON_TRANSPORT': 13, 'WALKING': 14, 'TRANSITION': 15}

    data['activityId'] = data['activity'].map(letter_to_number)

    undesired_activities = [3, 4, 9, 11, 13, 15]
    data = data[~data['activityId'].isin(undesired_activities)]
    # Transform the frequency of the data from 100Hz to 25Hz
    data = data.iloc[::4, :]

    columns_to_scale = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    scaler = RobustScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    size = len(data)
    train_data = data.iloc[0:int(size*0.8)]
    test_data = data.iloc[int(size*0.8):]

    X_train_sig, y_train = train_data[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']], train_data['activityId']
    X_test_sig, y_test = test_data[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']], test_data['activityId']

    if train_features:
        if not os.path.exists('domino_dataset/features'):
            os.mkdir('domino_dataset/features')

        cfg_file = tsfel.get_features_by_domain('statistical')
        X_train = tsfel.time_series_features_extractor(cfg_file, X_train_sig, fs=frequency, window_size=samples_required)
        X_test = tsfel.time_series_features_extractor(cfg_file, X_test_sig, fs=frequency, window_size=samples_required)
        X_train.to_csv(f'domino_dataset/features/X_train_comb_{frequency}.csv', index=False)
        X_test.to_csv(f'domino_dataset/features/X_test_comb_{frequency}.csv', index=False)
    else:
        X_train = pd.read_csv(f'domino_dataset/features/X_train_comb_{frequency}.csv')
        X_test = pd.read_csv(f'domino_dataset/features/X_test_comb_{frequency}.csv')

    X_train_columns = X_train.copy(deep=True)
    y_train = y_train[::samples_required]
    if len(y_train) > len(X_train):
        y_train = y_train.drop(y_train.tail(1).index)

    y_test = y_test[::samples_required]
    if len(y_test) > len(X_test):
        y_test = y_test.drop(y_test.tail(1).index)

    # Highly correlated features are removed
    corr_features = tsfel.correlated_features(X_train)
    X_train.drop(corr_features, axis=1, inplace=True)
    X_test.drop(corr_features, axis=1, inplace=True)

    # Remove low variance features
    selector = VarianceThreshold(threshold=0.5)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    # Get columns to keep and create new dataframe with those only
    cols_idxs = selector.get_support(indices=True)
    print('len cols idxs', len(cols_idxs))
    X_train_columns = X_train_columns.iloc[:, cols_idxs]
    print('Selected Features', *X_train_columns.columns)

    # Normalising Features
    scaler = preprocessing.StandardScaler()
    nX_train = scaler.fit_transform(X_train)
    nX_test = scaler.transform(X_test)

    return nX_train, y_train, nX_test, y_test


def create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name):
    model = keras.Sequential()
    if chosen_model == 'lstm_1':
        model.add(keras.layers.LSTM(units=64, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.3))
    elif chosen_model == 'gru_1':
        model.add(keras.layers.GRU(units=64, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.3))
    elif chosen_model == 'lstm_2':
        model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
        model.add(keras.layers.LSTM(units=32, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.3))
    elif chosen_model == 'gru_2':
        model.add(keras.layers.GRU(units=64, return_sequences=True, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
        model.add(keras.layers.GRU(units=32, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.3))
    elif chosen_model == 'cnn_lstm':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.LSTM(units=32, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == 'cnn_gru':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.GRU(units=32, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == 'cnn_cnn_lstm':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.LSTM(units=64, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == 'cnn_cnn_gru':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.GRU(units=64, return_sequences=False, input_shape=input_shape))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == 'cnn_cnn':
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(keras.layers.Dropout(rate=0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.4))
    elif chosen_model == '2cnn_2cnn':
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(keras.layers.Dropout(rate=0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.4))

    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # print(model.summary())
    model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.3, verbose=2)
    model.save(file_name)

    return model


def train_sequential_model(X_train, y_train, X_test, y_test, chosen_model, class_labels, train_model):
    if not os.path.exists('domino_dataset/models'):
        os.makedirs('domino_dataset/models')

    file_name = f'domino_dataset/models/comb_{chosen_model}_model.h5'
   
    if train_model:
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_sequential_model(X_train, y_train, chosen_model, input_shape, file_name)
    else:
        model = keras.models.load_model(file_name)

    loss, accuracy = model.evaluate(X_train, y_train)
    print("Train Accuracy: %d%%/ Train Loss: %d%%" % (100*accuracy, 100*loss))

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
    # loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy: %d%%" % (100*accuracy))
    print("Test F1 Score: %d%%" % (100*f1))

    report = classification_report(y_test_labels, y_pred_labels, target_names=class_labels)
    print(report)

    return y_test_labels, y_pred_labels


def train_feature_model(X_train, y_train, X_test, y_test, chosen_model, class_labels, train_model):
    if train_model:
        if chosen_model == 'rf':
            classifier = RandomForestClassifier(n_estimators=40, min_samples_split=15, min_samples_leaf=4, max_depth=None, bootstrap=True, n_jobs=-1, random_state=42)
            classifier.fit(X_train, y_train.ravel())
        elif chosen_model == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='uniform')
            classifier.fit(X_train, y_train.ravel())
        file = open(f'domino_dataset/models/comb_{chosen_model}_model.pkl', 'wb') 
        pickle.dump(classifier, file)
    else:
        file = open(f'domino_dataset/models/comb_{chosen_model}_model.pkl', 'rb') 
        classifier = pickle.load(file)

    classifier.fit(X_train, y_train.ravel())
    y_pred_train = classifier.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    print("Training Accuracy: %.2f%%" % (round(train_accuracy*100, 2)))

    y_test_predict = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_predict)
    f1 = f1_score(y_test, y_test_predict, average='weighted')
    print("Accuracy: %.2f%% / F1 Score: %.2f%%" % (100 * round(accuracy, 2), (100 * round(f1, 2))))
    report = classification_report(y_test, y_test_predict, target_names=class_labels)
    print(report)

    return y_test, y_test_predict


def plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, chosen_model):
    if not os.path.exists('domino_dataset/plots'):
        os.makedirs('domino_dataset/plots')

    normalize_cm = [None, 'true']
    for norm_value in normalize_cm:
        if norm_value == 'true':
            format = '.2f'
            plot_name = f'comb_{chosen_model}_cm_normalized.png'
        else:
            format = 'd'
            plot_name = f'comb_{chosen_model}_cm.png'

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test_labels, y_pred_labels,
            display_labels=class_labels,
            normalize=norm_value,
            xticks_rotation=70,
            values_format=format,
            cmap=plt.cm.Blues
        )

        plt.figure(figsize=(8, 10))
        plt.title(f'Confusion Matrix for {chosen_model}')
        disp.plot(cmap=plt.cm.Blues, values_format=format)
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'domino_dataset/plots/{plot_name}', bbox_inches='tight', pad_inches=0.1)
        # plt.show()
    

if __name__ == '__main__':
    frequency = 25
    time_required_ms = 3500
    samples_required = int(time_required_ms * frequency / 1000)

    path = "domino_dataset/data.csv"
    class_labels = ['Brushing teeth', 'Cycling', 'Lying', 'Moving by car', 'Running', 'Sitting', 'Stairs', 'Standing', 'Walking']

    # Choose the model
    models = ['lstm_1', 'gru_1', 'lstm_2', 'gru_2', 'cnn_lstm', 'cnn_gru', 'cnn_cnn_lstm', 'cnn_cnn_gru', 'cnn_cnn', '2cnn_2cnn', 'rf', 'knn']
    # chosen_model = models[0]
    X_train, y_train, X_test, y_test = split_data(path, samples_required)

    for chosen_model in models:
        print(f'{chosen_model=}')
        
        if chosen_model == 'knn' or chosen_model == 'rf':
            X_train, y_train, X_test, y_test = extract_features(path, frequency, samples_required, train_features=False)
            y_test_labels, y_pred_labels = train_feature_model(X_train, y_train, X_test, y_test, chosen_model, class_labels, train_model=False)
        else:
            # X_train, y_train, X_test, y_test = split_data(path, samples_required)
            y_test_labels, y_pred_labels = train_sequential_model(X_train, y_train, X_test, y_test, chosen_model, class_labels, train_model=False)

        plot_confusion_matrix(y_test_labels, y_pred_labels, class_labels, chosen_model)

