import data_preprocessing
import model
import hyperparameter_tuning

if __name__ == "__main__":
    training = "UNSW_NB15_training-set.csv"
    testing = "UNSW_NB15_testing-set.csv"
    
    x_train, y_train, x_test, y_test = data_preprocessing.read_data(training, testing)
    x_train = data_preprocessing.scale_data(x_train)
    x_test = data_preprocessing.scale_data(x_test)
    
    hyperparameter_tuning.hyperparameter_tuning(x_train, y_train)
    
    input_shape = (7, 38)
    units = 32
    epochs = 20
    batch_size = 32
    
    lstm_model = model.build_lstm_model(input_shape, units)
    train_time = model.train_model(lstm_model, x_train, y_train, epochs, batch_size)
    prec, recall, f_1, acc, myauc, mypr = model.evaluate_model(lstm_model, x_test, y_test)
    
    print("Train Time:", train_time)
    print("Precision:", prec)
    print("Recall:", recall)
    print("F1 Score:", f_1)
    print("Accuracy:", acc)
    print("AUC:", myauc)
    print("Average Precision:", mypr)
