from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier # <-- Import do XGBoost aqui
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
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
        # --- ATIVE A PROBABILIDADE AQUI ---
        ('svm', SVC(random_state=random_state, class_weight='balanced', probability=True))
    ])
    
    # Espaço de busca exato da Tabela 5 do artigo
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__gamma': ['scale', 'auto']
    }
    
    return pipeline, param_grid

# --- NOVO BLOCO DO XGBOOST ---
def get_xgboost(random_state=42):
    """Retorna um Pipeline e a grade de busca para o XGBoost."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            random_state=random_state, 
            eval_metric='mlogloss' # Evita warnings use_label_encoder=False
            
        ))
    ])
    
    # Espaço de busca clássico e otimizado para o XGBoost
    param_grid = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__max_depth': [3, 6, 10],
        'xgb__subsample': [0.8, 1.0] # Ajuda a evitar overfitting
    }
    
    return pipeline, param_grid
# -----------------------------

def train_and_evaluate(pipeline, param_grid, X_train, y_train, X_test, y_test):
    print(f"  -> Otimizando hiperparâmetros (GridSearch em andamento...)")
    
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring='f1_macro', cv=3, n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"     Melhores parâmetros: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    # --- NOVO: CÁLCULO SEGURO DO ROC-AUC MULTICLASSE ---
    try:
        y_proba = best_model.predict_proba(X_test)
        # Checa se é problema binário ou multiclasse
        if len(np.unique(y_train)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except Exception as e:
        print(f"     [Aviso] Não foi possível calcular ROC-AUC: {e}")
        roc_auc = 0.0
    # --------------------------------------------------

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"     Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | AUC: {roc_auc:.4f}")
    
    # Agora retorna 4 valores em vez de 3
    return bal_acc, macro_f1, roc_auc, grid_search.best_params_
