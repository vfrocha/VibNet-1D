from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

def get_random_forest(n_estimators=100, random_state=42):
    """Retorna uma Random Forest configurada."""
    # class_weight='balanced' ajuda muito com o desbalanceamento inerente de falhas
    return RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state, 
        class_weight='balanced',
        n_jobs=-1 # Usa todos os núcleos do processador para treinar mais rápido
    )

def get_svm(kernel='rbf', C=10.0, random_state=42):
    """Retorna um SVM configurado."""
    return SVC(
        kernel=kernel, 
        C=C, 
        random_state=random_state, 
        class_weight='balanced'
    )

def train_and_evaluate(model, X_train, y_train, X_test, y_test, target_names=None):
    """
    Padroniza os dados (Z-score), treina o modelo e calcula as métricas.
    """
    print(f"  -> Treinando {model.__class__.__name__}...")
    
    # 1. Normalização (Crucial para SVM e métricas de distância)
    # Ajustamos o scaler APENAS no treino para evitar vazamento de dados (data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Treinamento
    model.fit(X_train_scaled, y_train)
    
    # 3. Predição e Métricas
    y_pred = model.predict(X_test_scaled)
    
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"     Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f}")
    
    return bal_acc, macro_f1, y_pred
