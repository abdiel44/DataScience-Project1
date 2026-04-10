from pathlib import Path

import pandas as pd

from pre_processing.eda import run_eda


def test_run_eda_generates_expected_outputs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "eeg_band_alpha": [1.2, 1.5, 1.1, 0.9, 1.8, 1.0],
            "eeg_band_beta": [0.4, 0.55, 0.35, 0.2, 0.65, 0.3],
            "subject_group": ["A", "A", "B", "B", "A", "B"],
            "sleep_stage": ["N1", "N2", "N2", "REM", "N1", "REM"],
        }
    )
    output_dir = tmp_path / "eda_sleep"

    result = run_eda(
        df=df,
        output_dir=output_dir,
        task="sleep",
        target_col="sleep_stage",
        top_n=2,
    )

    expected_tables = {
        "profile": output_dir / "01_dataset_profile.csv",
        "numeric": output_dir / "02_descriptive_numeric.csv",
        "categorical": output_dir / "03_descriptive_categorical.csv",
        "pearson_corr": output_dir / "04_correlations_pearson.csv",
        "spearman_corr": output_dir / "05_correlations_spearman.csv",
    }

    for key, path in expected_tables.items():
        assert Path(result["tables"][key]).exists()
        assert path.exists()

    summary_path = output_dir / "eda_summary.md"
    assert Path(result["summary"]).exists()
    assert summary_path.exists()

    assert len(result["figures"]) >= 3
    for fig_path in result["figures"]:
        path = Path(fig_path)
        assert path.exists()
        assert path.suffix == ".png"
