import os
from src.experiments.engine_tabular import run_universal_tabnet

# Caminhos padrão do projeto
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# Parâmetros exclusivos da base PU (Paderborn University)
# Verifique se o nome exato da pasta dentro de 'data/processed' é 'PU' ou 'PU_Dataset'
DATASET = "PU" 

# As 4 condições operacionais clássicas da base PU (Velocidade_Torque_Força)
# IMPORTANTE: Verifique se os nomes das pastas no seu diretório são exatamente estes:
CONDITIONS = [
    "C1_1500rpm_0.7Nm_1000N",
    "C2_900rpm_0.7Nm_1000N",
    "C3_1500rpm_0.1Nm_1000N",
    "C4_1500rpm_0.7Nm_400N"
]

if __name__ == "__main__":
    run_universal_tabnet(
        dataset_name=DATASET,
        conditions=CONDITIONS,
        data_root=DATA_ROOT,
        results_dir=RESULTS_DIR
    )
