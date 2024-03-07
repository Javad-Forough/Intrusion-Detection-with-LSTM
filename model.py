import time
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def build_lstm_model(input_shape, units):
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, activation='tanh', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs, batch_size):
    # Train the model
    train_start = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    train_end = time.time()
    train_time = train_end - train_start
    return train_time

def evaluate_model(model, x_test, y_test):
    # Evaluate the model
    predictions = model.predict_classes(x_test, verbose=0)
    pred = predictions.flatten()
    
    prec = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f_1 = f1_score(y_test, pred)
    myauc = roc_auc_score(y_test, pred)
    mypr = average_precision_score(y_test, pred)
    acc = model.evaluate(x_test, y_test)[1]
    
    print("Precision = ", prec.__round__(4))
    print("Recall = ", recall.__round__(4))
    print("F1 = ", f_1.__round__(4))
    print("ACC = ", acc.__round__(4))
    print("AUC = ", myauc.__round__(4))
    print("PR = ", mypr.__round__(4))
    
    return prec, recall, f_1, acc, myauc, mypr
