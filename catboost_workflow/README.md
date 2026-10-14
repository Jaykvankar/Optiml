# CatBoost GPU workflow

This folder contains a separate Colab notebook for CatBoost. Existing LightGBM notebooks and run folders are untouched. It uses the shared `optiml` data preparation and training code, with CatBoost explicitly set to `--device GPU`.

1. In VS Code, open `catboost_colab.ipynb` and select **Colab → New Colab Server → GPU** as the kernel. Run its `nvidia-smi` cell. Your local PowerShell terminal will still use your laptop.
2. Right-click `source.zip` in VS Code Explorer and select **Upload to Colab**. The notebook expects `/content/source.zip`; change `SOURCE_ZIP` if the Colab Contents panel shows a different path.
3. Run source setup, CatBoost installation/preflight, Drive mounting and feature preparation, then the sampled smoke test.
4. Run the full depth-6 baseline. Compare its AUC and monthly AUC to the previous LightGBM validation result. Only then enable the CatBoost trials. These run one at a time and back up to `MyDrive/OptimlRuns/CatBoost/jay`.
5. Check the best models at the October cutoff. Set `FINALIZE=True` only after choosing a configuration. Audit its submission file and upload that exact CSV to Kaggle.

If shared `optiml` code changes, run `python catboost_workflow/build.py` locally and upload the rebuilt `source.zip`. A fresh Colab runtime or a new `REPO` path is needed for a changed ZIP.

The earlier 0.5656 leaderboard score is not explained by the CPU/GPU difference. When its exact submitted CSV and generating run are available, run the notebook audit cell.
