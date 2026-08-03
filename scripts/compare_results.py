import os
import pandas as pd
import numpy as np

# --- CONFIGURAÇÕES DE ARQUIVOS ---
# Ajuste os nomes dos arquivos para os mais recentes
BASELINE_CSV = "baseline_loco_results_20260728_192137.csv"
TL_CSV = "tl_results_20260803_071723.csv"
OUTPUT_CSV = "comparacao_tl_vs_baseline.csv"

# Diretório base onde os resultados costumam ficar salvos
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))

def format_variation(val):
    """Formata a variação com sinal de + ou - e um emoji de status."""
    if pd.isna(val):
        return "N/A"
    
    # Arredonda para 4 casas decimais
    val = round(val, 4)
    if val > 0.05:
        return f"+ {val:.4f} 🟢"
    elif val > 0:
        return f"+ {val:.4f} 🟡"
    else:
        return f"- {abs(val):.4f} 🔴"

def run_comparison():
    path_baseline = os.path.join(RESULTS_DIR, BASELINE_CSV)
    path_tl = os.path.join(RESULTS_DIR, TL_CSV)

    # Tenta carregar no diretório de resultados, se não achar, tenta no diretório atual
    if not os.path.exists(path_baseline): path_baseline = BASELINE_CSV
    if not os.path.exists(path_tl): path_tl = TL_CSV

    if not os.path.exists(path_baseline) or not os.path.exists(path_tl):
        print(f"[Erro] Certifique-se de que os arquivos:\n1) {BASELINE_CSV}\n2) {TL_CSV}\nestão na mesma pasta do script ou na pasta 'results'.")
        return

    print("Carregando arquivos de resultados...")
    df_base = pd.read_csv(path_baseline)
    df_tl = pd.read_csv(path_tl)

    # 1. Filtra apenas a tarefa de Detecção (Detection)
    df_base_det = df_base[df_base['Task'].str.lower() == 'detection'].copy()
    df_tl_det = df_tl[df_tl['Task'].str.lower() == 'detection'].copy()

    # 2. Agrega os resultados do Baseline (Média entre todos os Folds/Test Conditions)
    print("Agregando médias do Baseline LOCO...")
    base_agg = df_base_det.groupby(['Dataset', 'Model'])[['Bal Acc', 'Macro F1', 'ROC-AUC']].mean().reset_index()
    base_agg.rename(columns={
        'Dataset': 'Target Domain', 
        'Macro F1': 'Baseline F1',
        'Bal Acc': 'Baseline Acc'
    }, inplace=True)

    # 3. Agrega os resultados de Transfer Learning
    print("Extraindo resultados do Transfer Learning...")
    tl_agg = df_tl_det.groupby(['Target Domain', 'Model'])[['Bal Acc', 'Macro F1', 'ROC-AUC']].mean().reset_index()
    tl_agg.rename(columns={
        'Macro F1': 'TL F1',
        'Bal Acc': 'TL Acc'
    }, inplace=True)

    # 4. Mescla os dois DataFrames
    print("Cruzando os dados...")
    merged = pd.merge(base_agg, tl_agg, on=['Target Domain', 'Model'], how='outer')

    # Remove modelos que você não rodou no TL (como XGBoost ou TabNet se for o caso)
    merged = merged.dropna(subset=['TL F1'])

    # 5. Calcula a Variação do Macro F1 (TL - Baseline)
    merged['Variation (F1)'] = merged['TL F1'] - merged['Baseline F1']
    
    # Aplica a formatação visual na variação
    merged['Status'] = merged['Variation (F1)'].apply(format_variation)

    # Organiza a ordem das colunas para visualização
    final_columns = ['Target Domain', 'Model', 'Baseline F1', 'TL F1', 'Status']
    df_final = merged[final_columns].copy()

    # Preenche NaNs visuais (Ex: HUST que falhou no baseline)
    df_final['Baseline F1'] = df_final['Baseline F1'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    df_final['TL F1'] = df_final['TL F1'].apply(lambda x: f"{x:.4f}")

    # Ordena pelo Dataset alvo
    df_final.sort_values(by=['Target Domain', 'Model'], inplace=True)

    print("\n" + "="*80)
    print(" RELATÓRIO DE TRANSFER LEARNING VS BASELINE (MACRO F1 - DETECTION)")
    print("="*80)
    print(df_final.to_string(index=False))
    print("="*80)

    # Salva o resultado
    out_path = os.path.join(RESULTS_DIR, OUTPUT_CSV)
    df_final.to_csv(out_path, index=False)
    print(f"\n[Sucesso] Tabela comparativa salva em: {out_path}")

if __name__ == "__main__":
    run_comparison()
