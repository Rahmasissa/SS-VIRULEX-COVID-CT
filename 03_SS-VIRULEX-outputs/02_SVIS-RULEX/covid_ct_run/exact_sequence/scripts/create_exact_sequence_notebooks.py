from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


RUN_ROOT = Path("/Users/samehissa/Downloads/XAI-Med-Images-Stat-Visual-Rules-main 2/covid_ct_run/exact_sequence")
NOTEBOOK_DIR = RUN_ROOT / "notebooks"
SCRIPT_DIR = RUN_ROOT / "scripts"


SETUP = rf'''
import sys
from pathlib import Path

SCRIPT_DIR = Path("{SCRIPT_DIR}")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import *
setup_environment()
print("Run root:", RUN_ROOT)
print("Output dir:", OUTPUT_DIR)
'''


def write_notebook(path: Path, title: str, body_cells: list[str]) -> None:
    cells = [new_markdown_cell(f"# {title}"), new_code_cell(SETUP)]
    cells.extend(new_code_cell(cell) for cell in body_cells)
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "COVID-CT venv", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbformat.write(notebook, path)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    write_notebook(
        NOTEBOOK_DIR / "01_image_augmentation_code_covid_ct.ipynb",
        "01 - Image Augmentation Code on COVID-CT",
        [
            '''
official = build_official_split_table()
print("Official split counts:")
print(official.groupby(["split", "class_name"]).size().unstack(fill_value=0))
''',
            '''
augmented = create_augmented_manifest()
print("Augmented split counts:")
print(augmented.groupby(["split", "class_name", "source"]).size())
print("Saved:", OUTPUT_DIR / "01_augmented_splits.csv")
''',
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "02_dl_model_hyperparameter_optimization_using_grid_search_covid_ct.ipynb",
        "02 - DL Model Hyperparameter Optimization Using Grid Search",
        [
            '''
augmented = load_augmented_splits()
print("Input split counts:")
print(augmented.groupby(["split", "class_name"]).size().unstack(fill_value=0))
''',
            '''
results, best_params = run_dl_grid_search()
print("Best hyperparameters:")
print(best_params)
print(results.to_string(index=False))
''',
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "03_custom_mobilenetv2_complete_code_covid_ct.ipynb",
        "03 - Custom MobileNetV2 Complete Code on COVID-CT",
        [
            '''
with (OUTPUT_DIR / "02_best_dl_hyperparameters.json").open("r", encoding="utf-8") as handle:
    print("Best hyperparameters from grid search:")
    print(json.load(handle))
''',
            '''
metrics = train_final_custom_mobilenet()
print("Final custom MobileNetV2 metrics:")
print(json.dumps({k: metrics[k] for k in ["accuracy", "balanced_accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc", "confusion_matrix"]}, indent=2))
''',
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "04_statistical_features_extraction_and_feature_selection_using_zfmis_covid_ct.ipynb",
        "04 - Statistical Features Extraction and Feature Selection Using ZFMIS",
        [
            '''
feature_sets = run_statistical_features_zfmis()
for name, columns in feature_sets.items():
    print(name, len(columns), columns)
print("Saved feature sets:", FEATURE_DIR / "04_zfmis_feature_sets.json")
''',
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "05a_gridsearchfortree_covid_ct.ipynb",
        "05a - GridSearchForTree on COVID-CT Statistical Features",
        [
            '''
results = evaluate_tree_grid()
print(results.to_string(index=False))
''',
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "05b_rulefitgridsearchcode_covid_ct.ipynb",
        "05b - RuleFit GridSearch Code on COVID-CT Statistical Features",
        [
            '''
results = evaluate_rulefit_grid()
print(results.to_string(index=False))
''',
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "06_rule_extraction_code_covid_ct.ipynb",
        "06 - Rule Extraction Code on COVID-CT",
        [
            '''
summary = extract_final_rules()
print(json.dumps(summary, indent=2))
''',
            '''
metric_summary = final_sequence_summary()
print(metric_summary.to_string(index=False))
''',
        ],
    )

    write_notebook(
        NOTEBOOK_DIR / "07_sfmov_heatmaps_covid_ct.ipynb",
        "07 - SFMOV Heatmaps on COVID-CT",
        [
            '''
saved = run_sfmov_heatmaps(samples_per_class=4)
print("Saved heatmap files:")
for path in saved:
    print(path)
''',
            '''
metric_summary = final_sequence_summary()
print(metric_summary.to_string(index=False))
''',
        ],
    )

    print("Created exact-sequence notebooks:")
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        print(path)


if __name__ == "__main__":
    main()
