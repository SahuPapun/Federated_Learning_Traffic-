import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector
from keras.optimizers import Adam
from keras.losses import Loss, Huber
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from PythonUtils import *
from tensorflow.python.keras.models import load_model

def get_weighted_average_of_model_and_set_weight_of_server_model_and_save_server_weights(w_1,w_2,w_3,w_4,w_5):
    c_path = os.getcwd()
    server_space = os.path.join(c_path, 'data', 'server_space')
    model1_weight_path = os.path.join(server_space, 'client1_weights.h5')
    model2_weight_path = os.path.join(server_space, 'client2_weights.h5')
    model3_weight_path = os.path.join(server_space, 'client3_weights.h5')
    model4_weight_path = os.path.join(server_space, 'client4_weights.h5')
    model5_weight_path = os.path.join(server_space, 'client5_weights.h5')

    server_model_path = os.path.join(server_space, 'server_model.h5')
    m1 = load_model(server_model_path)
    m2 = load_model(server_model_path)
    m3 = load_model(server_model_path)
    m4 = load_model(server_model_path)
    m5 = load_model(server_model_path)

    m1.load_weights(model1_weight_path)
    m2.load_weights(model2_weight_path)
    m3.load_weights(model3_weight_path)
    m4.load_weights(model4_weight_path)
    m5.load_weights(model5_weight_path)
    # print(w_1,w_2,w_3,w_4,w_5)
    weights_model1 = m1.get_weights()
    weights_model2 = m2.get_weights()
    weights_model3 = m3.get_weights()
    weights_model4 = m4.get_weights()
    weights_model5 = m5.get_weights()
    average_weights = [(w_1 * w1 + w_2 * w2 + w_3 * w3 + w_4 * w4 + w_5 * w5) for w1, w2, w3, w4, w5 in zip(weights_model1, weights_model2, weights_model3, weights_model4, weights_model5)]

    delete_file(model1_weight_path)
    delete_file(model2_weight_path)
    delete_file(model3_weight_path)
    delete_file(model4_weight_path)
    delete_file(model5_weight_path)
    m = load_model(server_model_path)
    m.set_weights(average_weights)
    m.save_weights(os.path.join(server_space, 'server_weights.h5'))

def load_server_model_and_set_weights(model,average_weights):
    c_path = os.getcwd()
    server_space = os.path.join(c_path, 'data', 'server_space')
    model_path = os.path.join(server_space, model)
    m = load_model(model_path)
    m.set_weights(average_weights)

def initialize_global_model_01(model_name):
    c_path = os.getcwd()
    model = create_global_model_03()
    dummy_input = np.random.rand(1, 1, 70).astype(np.float32)
    model(dummy_input)
    model.save(os.path.join(c_path, 'data', 'server_space', model_name))

def create_global_model_01():
    model = Sequential()
    # Add LSTM
    model.add(LSTM(5, input_shape=(1, 70)))
    model.add(Dense(1))
    opt = Adam(learning_rate=0.1)
    model.compile(optimizer=opt, loss=Huber(), metrics=["mse"])
    return model

def create_global_model_02():
    ts_model = Sequential()
    # Add LSTM
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dropout(0.2))
    ts_model.add(RepeatVector(30))
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dropout(0.2))
    ts_model.add(RepeatVector(30))
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dense(1))
    opt = Adam(learning_rate=.0001)
    # Compile with Adam Optimizer. Optimize for minimum mean square error
    ts_model.compile(optimizer=opt, loss=Huber(), metrics=["mse"])
    return ts_model

def create_global_model_03():
    ts_model = Sequential()
    # Add LSTM
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dropout(0.2))
    ts_model.add(RepeatVector(30))
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dropout(0.2))
    ts_model.add(RepeatVector(30))
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dense(1))
    opt = Adam(learning_rate=.0001)
    # Compile with Adam Optimizer. Optimize for minimum mean square error
    ts_model.compile(optimizer=opt, loss='mean_squared_error', metrics=["mse"])
    return ts_model

def create_global_model_04():
    ts_model = Sequential()
    # Add LSTM
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dropout(0.2))
    ts_model.add(RepeatVector(30))
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dropout(0.2))
    ts_model.add(RepeatVector(30))
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dropout(0.2))
    ts_model.add(RepeatVector(30))
    ts_model.add(LSTM(30, input_shape=(1, 70)))
    ts_model.add(Dense(1))
    opt = Adam(learning_rate=.0001)
    # Compile with Adam Optimizer. Optimize for minimum mean square error
    ts_model.compile(optimizer=opt, loss='mean_squared_error', metrics=["mse"])
    return ts_model

def train_model_without_callback(ts_model, epochs, batch_size,train_x, train_y,val_x, val_y):
    history = ts_model.fit(train_x, train_y,
                           epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(val_x, val_y),
                           shuffle=False)
    return history

def update_loss(trainHistoryDict, history,loss_name):
    file_path = trainHistoryDict+ '_'+loss_name
    if os.path.exists(file_path):
        with open(file_path, "rb") as file_pi:
            loss = pickle.load(file_pi)
        with open(file_path, 'wb') as file_pi:
            pickle.dump(loss + history.history[loss_name], file_pi)
    else:
        with open(file_path, 'wb') as file_pi:
            pickle.dump(history.history[loss_name], file_pi)

def update_loss_n_val_loss(trainHistoryDict, history,loss_name):
    update_loss(trainHistoryDict, history, loss_name)
    update_loss(trainHistoryDict, history, 'val_' + loss_name)


def load_test_n_validation_losses(trainHistoryDict, loss_name):
    with open(trainHistoryDict + '_'+loss_name, "rb") as file_pi:
        loss = pickle.load(file_pi)
    with open(trainHistoryDict + '_val_'+loss_name, "rb") as file_pi:
        val_loss = pickle.load(file_pi)
    return loss,val_loss

def plot_train_and_test_loss_wrt_epoch(path,client,y1, y2):
    plt.plot(list(y1))
    plt.plot(list(y2))
    plt.title('Learning Curve'+'('+client+')')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    # plt.legend(['Training loss: '+str(list(y1)[-1]), 'Validation loss: '+str(list(y2)[-1])], loc='upper right')
    plt.savefig(os.path.join(path, "LearningCurve.jpg"))
    # print('Training loss: '+str(list(y1)[-1]), 'Validation loss: '+str(list(y2)[-1]))

def update_and_save_the_learning_curve(path,training_history,loss_name,client):
    history_path = os.path.join(path, 'history')
    update_loss_n_val_loss(history_path, training_history, loss_name)
    t, v = load_test_n_validation_losses(history_path, loss_name)
    plot_train_and_test_loss_wrt_epoch(path,client, t, v)

def create_rnn_dataset(data, lookback=1):
    data_x, data_y = [], []
    for i in range(len(data) - lookback - 1):
        a = data[i:(i + lookback), 0]
        data_x.append(a)
        # The next point
        data_y.append(data[i + lookback, 0])
    return np.array(data_x), np.array(data_y)

def load_data(path,col_name):
    np.random.seed(1)
    df = pd.read_csv(os.path.join(path, col_name + ".csv"), index_col='source')
    df = df[[col_name]]
    # print(col_name,': ',len(df))
    # print('Std: ' + str(df.std()))
    df.sort_index(inplace=True)
    return df

def make_rnn_data(col_name,df,train_portion,lookback):
    scaler = StandardScaler()
    scaled_df=scaler.fit_transform(df)
    train_size = int(len(scaled_df)*train_portion)
    train_df = scaled_df[0:train_size,:]
    test_df = scaled_df[train_size-lookback:,:]
    # print("\nShaped of Train, Test(",col_name,") : ", train_df.shape, test_df.shape)
    train_x, train_y = create_rnn_dataset(train_df,lookback)
    train_x = np.reshape(train_x, (train_x.shape[0],1, train_x.shape[1]))
    # print("Shapes of X, Y(train)(",col_name,") : ", train_x.shape, train_y.shape)
    test_x, test_y = create_rnn_dataset(test_df,lookback)
    test_x = np.reshape(test_x, (test_x.shape[0],1, test_x.shape[1]))
    # print("Shapes of X, Y(val+test)(",col_name,") : ", test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y

def find_mean_squared_error(model,client,y_test,x_test):
    # pass
    y_pred = model.predict(x_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(client+"- Mean Squared Error: "+str(mse))

def save_scaler_to_disk(scaler, path):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler_from_disk(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict_on_new_data(model, data_values, lookback=70):
    scaler = StandardScaler()
    data_array = np.array(data_values).reshape(-1, 1)
    scaled = scaler.fit_transform(data_array)
    x_seq, y_seq = create_rnn_dataset(scaled, lookback)
    if len(x_seq) == 0:
        return None, None, scaler
    x_input = np.reshape(x_seq, (x_seq.shape[0], 1, x_seq.shape[1]))
    y_pred_scaled = model.predict(x_input).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()
    return y_actual.tolist(), y_pred.tolist(), scaler

def format_inference_results(actual, predicted):
    results = []
    for i, (a, p) in enumerate(zip(actual, predicted)):
        results.append({
            'index': i,
            'actual': round(float(a), 4),
            'predicted': round(float(p), 4),
            'error': round(float(a - p), 4)
        })
    return results
