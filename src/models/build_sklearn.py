from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def get_random_forest(random_state=42):
    """Retorna um Pipeline e a grade de busca para a Random Forest."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=random_state, class_weight='balanced'))
    ])
    
    # Espaço de busca exato da Tabela 5 do artigo
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__criterion': ['gini', 'entropy', 'log_loss'],
        'rf__max_depth': [10, 25, 50, None],
        'rf__min_samples_split': [2, 5, 10]
    }
    
    return pipeline, param_grid

def get_svm(random_state=42):
    """Retorna um Pipeline e a grade de busca para o SVM."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=random_state, class_weight='balanced'))
    ])
    
    # Espaço de busca exato da Tabela 5 do artigo
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__gamma': ['scale', 'auto']
    }
    
    return pipeline, param_grid

def train_and_evaluate(pipeline, param_grid, X_train, y_train, X_test, y_test):
    """
    Roda a validação aninhada (Grid Search) para encontrar os hiperparâmetros
    ideais no conjunto de treino, e depois avalia no conjunto de teste inédito.
    """
    print(f"  -> Otimizando hiperparâmetros (GridSearch em andamento...)")
    
    # Configura o GridSearch (cv=3 dobras internas para acelerar, otimizando o Macro F1)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1, # Usa todos os núcleos da CPU
        verbose=0
    )
    
    # 1. Ajusta o GridSearch (Ele testa todas as combinações no X_train)
    grid_search.fit(X_train, y_train)
    
    # Extrai o melhor modelo já treinado
    best_model = grid_search.best_estimator_
    print(f"     Melhores parâmetros: {grid_search.best_params_}")
    
    # 2. Predição Final no conjunto de Teste isolado
    y_pred = best_model.predict(X_test)
    
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"     Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f}")
    
    return bal_acc, macro_f1, grid_search.best_params_
