# FEED: Automated feedback generation for student answers (Operating Systems)

Code and results for the paper "Automated Feedback Generation for Student Responses: Comparative Analysis of Fine-tuned T5 and GPT-2".

## Data
- `OS-CUM.xlsx` is the only data file used for training and evaluation (student answers plus ChatGPT adversarial answers, with LLM-drafted feedback, a random 10% verified).
- Other `OS-*.xlsx` files, `Swetha.xlsx`, `Copy_of_gpt2_80_data.ipynb`, `code feedback_G.ipynb`, `Examples_on_how_to_run_it.ipynb` and the `Updated` folder are earlier material. They are NOT used for the reported results.

## Reproduce
Run `FEED_rerun_kaggle_github.ipynb` (Kaggle, GPU) or `feed_rerun.py`. Presets: `paper_random`, `paper_question`, `curve`.

## Results files
- `split_random.csv`, `split_question.csv` (+ `*_info.json`): row IDs with train/val/test assignment.
- `predictions_*`, `perex_*`: generated feedback and per-item scores for every system and seed.
- `random_*`, `question_*`, `curve_*`: results tables and training logs of the three runs (`results_all`, `training_log`, `paired_t5_minus_gpt2`, `table_main_mean_std_over_seeds`).
- `data_stats.json`: cleaning counts (98,656 raw rows -> 65,302 unique triples, 173 questions).
- `audit_tools.py`, `RATER_INSTRUCTIONS.md`: reference audit and blinded human-evaluation tools.
