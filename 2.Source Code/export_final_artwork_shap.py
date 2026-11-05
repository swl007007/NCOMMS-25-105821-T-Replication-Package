"""Export Figure 2 numbers using the original notebook computations.

Run in the frozen Windows Python 3.11.3 / XGBoost 2.0.3 environment, with
shap==0.46.0, numba==0.60.0 and llvmlite==0.43.0 installed.
Only the training and aggregation cells run; historical outputs are untouched.
"""

import ast
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "writing" / "final_artwork" / "data"


def main():
    expected = {"numpy": "1.26.4", "pandas": "2.2.3", "scikit-learn": "1.5.2",
                "xgboost": "2.0.3", "shap": "0.46.0"}
    assert platform.python_version() == "3.11.3" and platform.system() == "Windows"
    assert {name: version(name) for name in expected} == expected
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(HERE)
    provenance = {"python": platform.python_version(), "platform": platform.platform(),
                  "packages": {n: version(n) for n in [*expected, "scipy", "numba", "llvmlite"]},
                  "aggregation": "absolute signed sum over training observations, not mean absolute SHAP",
                  "sources": {}, "notebooks": {}}
    for label, filename, group_cell, raw_name in [
        ("forecasting", "Figure2_Feature_Importance_Forecasting.ipynb", 4, "shape_values_df_ensemble"),
        ("nowcasting", "Figure2_Nowcasting_two_layer_feature_importance.ipynb", 5, "shape_values_df_ensemble_L2"),
    ]:
        path = HERE / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        tree = ast.parse("".join(notebook["cells"][1]["source"]))
        start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.With))
        stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.For))
        statements = tree.body[start:stop + 1]
        assert isinstance(statements[-1].iter, ast.Call)
        assert ast.unparse(statements[-1].iter) == "range(2, 6)"
        scope = {"np": np, "pd": pd, "xgb": xgb, "shap": shap, "json": json}
        print(f"Computing original {label} SHAP values", flush=True)
        exec(compile(ast.Module(body=statements, type_ignores=[]), filename, "exec"), scope)
        raw = scope[raw_name]
        assert raw.groupby("phase").size().to_dict() == {2: 4405, 3: 4405, 4: 4405, 5: 4405}
        assert np.isfinite(raw.to_numpy()).all()
        exec(compile("".join(notebook["cells"][group_cell]["source"]), filename, "exec"), scope)
        signed = scope["shap_values_df_ensemble_sum"]
        grouped = scope["shap_values_df_ensemble_sum_grouped"]
        assert np.allclose(grouped.sum(axis=1), 1)
        signed.rename_axis("phase").to_csv(OUT / f"shap_{label}_signed_sums.csv")
        grouped.rename_axis("phase").to_csv(OUT / f"shap_{label}_groups.csv")
        rows = []
        for phase in (3, 4):
            top = signed.loc[phase].abs().nlargest(10)
            labels = scope["rename_shap_feature_names"](signed.abs(), list(signed.index).index(phase))
            assert np.allclose(top.to_numpy(), labels.to_numpy())
            rows.extend({"phase": phase, "rank": j, "feature": feature, "label": name,
                         "absolute_signed_sum": float(value)}
                        for j, (feature, name, value) in enumerate(zip(top.index, labels.index, top), 1))
        pd.DataFrame(rows).to_csv(OUT / f"shap_{label}_top10.csv", index=False)
        provenance["notebooks"][label] = {"path": filename, "training_cell_zero_based": 1,
            "aggregation_cell_zero_based": group_cell, "training_rows_per_phase": 4405,
            "features": len(raw.columns) - 1, "layer": 2 if label == "nowcasting" else 1}
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    sources = [HERE / n["path"] for n in provenance["notebooks"].values()]
    sources += [HERE / n for n in ("forecasting_hyperparameters.json", "forecasting_hyperparameters_p3.json")]
    sources += [HERE.parent / "1.Source Data" / f"{n}_Analysis_010825.csv" for n in ("Forecasting", "Nowcasting")]
    for path in sources:
        provenance["sources"][path.relative_to(HERE.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "shap_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
