# ZSSD through Disentangled Attention: A DeBERTa v3
This repository includes code for the DeBERTa v3 model , finetuned on EZSTANCE Dataset
**Text Preprocessing for ezStance (Subtask A)**

This script preprocesses tweet data from the `raw_train_all_onecol.csv` file for stance detection tasks.

## 🔄 Preprocessing Steps

1. Original tweets saved to a new column: `'Ori Text'`.
2. Restored missing apostrophes (e.g., `don t` → `don't`).
3. Expanded contractions (e.g., `can't` → `cannot`).
4. Removed URLs, emojis, special characters, @mentions, and hashtags.
5. Normalized possessives (`women's` → `women`) and common slang.
6. Split hashtags/mentions using `wordninja`.
7. Lowercased and cleaned tokens for modeling.

# Subtask A - Target Based ZSSD

## Overview
This repository implements a ZSSD model for Subtask A using `microsoft/deberta-v3-base`. It focuses on classifying stances (`FAVOR`, `AGAINST`, `NONE`) based on a combination of `Text` and `Target`.
## Experimentation Setup
 Target-based ZSSD (subtask A) aims to evaluate the classifier on a large number of completely unseen targets. We train models using three scenarios:1) on the full training set with both noun-phrase and claim targets; 2) on training data with noun-phrase targets only; and 3) training data with claim targets only. Each model is then evaluated in three corresponding scenarios: 1) the full test set with mixed targets; 2) the test subset with noun-phrase targets only; and 3) the test subset with claim targets only.
## Training
- Dataset is loaded using a custom `StanceDataset` class that tokenizes (text, target) pairs.
- The `DebertaClassifier` computes separate representations for text and topic by adjusting attention masks.
- Concatenated embeddings are passed through linear layers for final classification.

## Evaluation
- The model is evaluated on three test domains: `claim`, `noun_phrase`, and `mixed`.
- Accuracy, macro F1, and weighted F1 scores are reported.
- A full classification report and predictions are saved for each test set.

## Output
- Trained model saved at: `./saved_model`
- Classification reports and predictions saved in: `./results`

# Subtask B - Domain based ZSSD

## Overview
This repository implements domain generalization for ZSSD using the `microsoft/deberta-v3-base` model. The goal is to generalize across domains by training on 7 domains and testing on the held-out 8th.
## Experimentation Setup
Models trained on the full mixed-target dataset are evaluated across three settings: 1)the full mixed-target test set; 2)the noun phrase target-only test set; and 3)the claim target only test set, denoted as M, N, and C, respectively.
## Key Features
- **Model:** A custom `DebertaClassifier` that separately encodes `Text` and `Target` representations using attention masks.
- **Training Strategy:** Uses only entries where `In Use == 1` to avoid target leakage across domains.
- **Dataset:** For each domain, `raw_train_all_onecol.csv` and `raw_val_all_onecol.csv` are used. Test files are evaluated separately.
- **Evaluation:** Computes Accuracy, Macro F1, and Weighted F1. Classification reports and predictions are saved for all test domains.

## Output
- Model checkpoint: `./saved_model`
- Reports & predictions: `./results/`

## Notes
Ensure proper folder structure for subtask A and subtask B. During training, overlapping targets with the test domain are masked via the `In Use` column.




