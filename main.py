import data_preprocessing
import model

if __name__ == "__main__":
    train_file = "Benchmark/UNSW_NB15_training-set.csv"
    test_file = "Benchmark/UNSW_NB15_testing-set.csv"
    
    x_train, y_train, x_test, y_test = data_preprocessing.read_data(train_file, test_file)
    x_train = data_preprocessing.scale_data(x_train)
    x_test = data_preprocessing.scale_data(x_test)
    
    input_shape = (7, 38)
    units = 32
    epochs = 20
    batch_size = 32
    
    lstm_model = model.build_lstm_model(input_shape, units)
    model.train_model(lstm_model, x_train, y_train, epochs, batch_size)
    model.evaluate_model(lstm_model, x_test, y_test)
