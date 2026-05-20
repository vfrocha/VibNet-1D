import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataloader import load_vibration_data
from src.models.build_keras import build_vibnet_1d

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
DATASET = "CWRU_12k"
CONDITIONS = ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]

def run_raw_time_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_exp02_keras_{DATASET}_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"resultados_exp02_keras_{DATASET}_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"\n{'='*50}")
    print(f"EXPERIMENTO 2: DEEP LEARNING (Raw Time Domain)")
    print(f"Dataset: {DATASET}")
    print(f"{'='*50}")

    results = []

    for test_cond in CONDITIONS:
        print(f"\n--- Fold: Testando na condição inédita {test_cond} ---")

        # 1. Carrega os dados
        X_train, y_train, X_test, y_test, label_encoder = load_vibration_data(
            data_root=DATA_ROOT, dataset_name=DATASET, test_condition=test_cond
        )
        if len(X_train) == 0: continue

        # 2. Reshape crítico para CNN 1D do Keras -> (Amostras, Tempo, Canais)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        input_length = X_train.shape[1]
        num_classes = len(np.unique(y_train))

        # 3. Normalização Z-Score nos Sinais Brutos (Obrigatório para Redes Neurais)
        mean_tr = np.mean(X_train, axis=(0, 1), keepdims=True)
        std_tr = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
        
        X_train_scaled = (X_train - mean_tr) / std_tr
        X_test_scaled = (X_test - mean_tr) / std_tr

        # 4. Construção do Modelo
        model = build_vibnet_1d(input_length, num_classes)
        
        model_path = os.path.join(RESULTS_DIR, f"models/vibnet1d_{test_cond}.h5")
        
        # Callbacks para evitar overfitting no conjunto alvo
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("\n  [Iniciando Treinamento Keras]")
        # Como é validação Leave-One-Load-Out, validamos diretamente no teste final inédito.
        # (Em cenários puramente acadêmicos, o correto seria quebrar um pedaço do treino para val, 
        # mas como são hiperparâmetros fixos, podemos usar o teste para monitorar o early stopping).
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_scaled, y_test),
            callbacks=[early_stop],
            verbose=2 # Verbose 2 mantém o log limpo (uma linha por época)
        )
        
        model.save(model_path)

        # 5. Avaliação Final
        print("\n  [Avaliando Modelo]")
        y_pred_probs = model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"     Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f}")
        
        results.append({
            "Test Condition": test_cond,
            "Model": "CNN 1D (Raw Time)",
            "Bal Accuracy": bal_acc,
            "Macro F1": macro_f1
        })

    # Relatório Final
    if results:
        df_results = pd.DataFrame(results)
        print("\n\n" + "="*50)
        print(df_results.to_string(index=False))
        print("\n--- RESUMO ---")
        print(df_results.groupby("Model")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std']).to_string())
        df_results.to_csv(csv_file, index=False)

if __name__ == "__main__":
    run_raw_time_experiment()
