import os
from src.experiments.engine_tabular import run_universal_tabnet

# Caminhos padrão do projeto
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# Parâmetros exclusivos da base HUST
DATASET = "HUST" 
CONDITIONS = ["Load_0W", "Load_200W", "Load_400W"]

if __name__ == "__main__":
    run_universal_tabnet(
        dataset_name=DATASET,
        conditions=CONDITIONS,
        data_root=DATA_ROOT,
        results_dir=RESULTS_DIR
    )
