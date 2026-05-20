from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_vibnet_1d(input_length, num_classes):
    """
    Constrói a arquitetura CNN 1D para a Abordagem Raw Signal (Tempo).
    
    Parâmetros:
    - input_length: Tamanho da janela do sinal (ex: 12000)
    - num_classes: Quantidade de classes para classificação (softmax final)
    """
    model = Sequential(name="VibNet_1D_Raw")

    # Bloco 1: Kernel longo para capturar a macro-estrutura do sinal
    model.add(Conv1D(filters=32, kernel_size=64, strides=8, activation='relu', input_shape=(input_length, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Bloco 2: Kernel médio
    model.add(Conv1D(filters=64, kernel_size=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Bloco 3: Kernel curto para detalhes de alta frequência (impactos rápidos)
    model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Bloco 4: Extração profunda
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Transformação para fully-connected
    model.add(Flatten())
    
    # MLP Final (Classificador)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5)) # Regularização pesada
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizador configurado (learning rate padrão de 1e-3 costuma ser excelente)
    optimizer = Adam(learning_rate=0.001)
    
    # Como os rótulos do dataloader já são inteiros (LabelEncoder), usamos sparse_categorical
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
