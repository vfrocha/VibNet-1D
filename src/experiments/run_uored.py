import os
from src.experiments.engine_tabular import run_universal_tabnet

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

DATASET = "UORED" # Ou o nome da sua pasta da UORED

# --- O SEU DESIGN DE GRUPOS VIRTUAIS ---
# O motor agora entende que "Group_A" é o nome do Fold, 
# e a lista são as pastas que devem ser separadas para teste!
VIRTUAL_CONDITIONS = {
    "Group_A": ["Bearing_1", "Bearing_6", "Bearing_11", "Bearing_16"],
    "Group_B": ["Bearing_2", "Bearing_7", "Bearing_12", "Bearing_17"],
    "Group_C": ["Bearing_3", "Bearing_8", "Bearing_13", "Bearing_18"],
    "Group_D": ["Bearing_4", "Bearing_9", "Bearing_14", "Bearing_19"],
    "Group_E": ["Bearing_5", "Bearing_10", "Bearing_15", "Bearing_20"]
}

if __name__ == "__main__":
    run_universal_tabnet(
        dataset_name=DATASET,
        conditions=VIRTUAL_CONDITIONS,
        data_root=DATA_ROOT,
        results_dir=RESULTS_DIR
    )
