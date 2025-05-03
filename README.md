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
This repository implements a stance detection model for SemEval-style Subtask A using `microsoft/deberta-v3-base`. It focuses on classifying stances (`FAVOR`, `AGAINST`, `NONE`) based on a combination of `Text` and `Target`.

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
This repository implements domain generalization for stance detection using the `microsoft/deberta-v3-base` model. The goal is to generalize across domains by training on 7 domains and testing on the held-out 8th.

## Key Features
- **Model:** A custom `DebertaClassifier` that separately encodes `Text` and `Target` representations using attention masks.
- **Training Strategy:** Uses only entries where `In Use == 1` to avoid target leakage across domains.
- **Dataset:** For each domain, `raw_train_all_onecol.csv` and `raw_val_all_onecol.csv` are used. Test files are evaluated separately.
- **Evaluation:** Computes Accuracy, Macro F1, and Weighted F1. Classification reports and predictions are saved for all test domains.

## Output
- Model checkpoint: `./saved_model`
- Reports & predictions: `./results/`

## Notes
Ensure proper domain folder structure under `subtaskB/`. During training, overlapping targets with the test domain are masked via the `In Use` column.





3. For the 'subtaskB' train and val sets, we need to filter the 'In Use' column during actual training, using only the entries where 'In Use' == 1. This is because, during data splitting for subtaskB, one domain is chosen as the test set, and the remaining seven domains may have targets that overlap with the test domain. These overlapping targets need to be masked. The 'in use' column is used for this masking.

4. In our paper, we proposed a method that uses MNLI pre-trained models for training. In this case, the noun phrases are prompted (which also affects 'mixed' because it also contains noun-phrase targets). Therefore, in the paths corresponding to 'noun phrase' and 'mixed', there is a 'prompt' folder that contains the data with the prompted noun phrase targets. Results based on NLI prompting (e.g., **BART-MNLI-ep**) were obtained using these data.
