from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import model

def create_model(units):
    # Create LSTM model for grid search
    model = Sequential()
    model.add(LSTM(units, input_shape=(7, 38), activation='tanh', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def hyperparameter_tuning(x_train, y_train):
    # Perform hyperparameter tuning using grid search
    units = [16, 32, 64]
    batch_size = [16, 32, 64]
    epochs = [10, 20, 30]
    
    model = KerasClassifier(build_fn=create_model, verbose=0)
    param_grid = dict(units=units, batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
