from pathlib import Path

SAMPLE_SIZES = [1000, 5000, 25000]

TREE_DEPTHS = {
    1000: list(range(1,5)) + [6] + [8] + [10] + [15] + [20] + [25],
    5000: list(range(1,5)) + [6] + [8] + [10] + [15] + [20] + [30],
    25000: list(range(1,5)) + [6] + [8] + [10] + [15] + [20] + [35]
}

SPLITTING_TYPES = ["adaptive","honest"]

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

PLOTS_DIR = BASE_DIR / "plots"

TABLES_DIR = BASE_DIR / "tables"

for dir_path in [RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)




