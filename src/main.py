# Importando bibliotecas
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carregando o dataset Boston Housing do TensorFlow
from tensorflow.keras.datasets import boston_housing

def load_and_preprocess_data():
    # Carregando os dados
    (X_train, y_train), (X_test, y_test) = boston_housing.load_data()

    # Normalizando os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def build_model(input_shape):
    # Definindo a estrutura do modelo
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # Compilando o modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_history(history):
    # Plotando o histórico de treinamento
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.legend()
    plt.show()

def main():
    # Carregando e pré-processando os dados
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()

    # Construindo o modelo
    model = build_model(X_train_scaled.shape[1])

    # Treinando o modelo
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_split=0.2)

    # Avaliando o modelo
    loss = model.evaluate(X_test_scaled, y_test)
    print(f'Erro Quadrático Médio (MSE) no conjunto de teste: {loss}')

    # Plotando o histórico de treinamento
    plot_history(history)

    # Fazendo previsões
    y_pred = model.predict(X_test_scaled)

    # Plotando os valores reais vs. preditos
    plt.scatter(y_test, y_pred)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('Valores Reais vs. Preditos')
    plt.show()

    # Calculando o MSE manualmente
    mse = mean_squared_error(y_test, y_pred)
    print(f'Erro Quadrático Médio (MSE) calculado manualmente: {mse}')

if __name__ == '__main__':
    main()
