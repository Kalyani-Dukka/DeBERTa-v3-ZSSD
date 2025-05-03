# ZSSD through Disentangled Attention: A DeBERTa v3
This repository includes code for the DeBERTa v3 model , finetuned on EZSTANCE Dataset

# Text Preprocessing for ezStance (Subtask A)

This script preprocesses tweet data from the `raw_train_all_onecol.csv` file for stance detection tasks.

## 🔄 Preprocessing Steps

1. Original tweets saved to a new column: `'Ori Text'`.
2. Restored missing apostrophes (e.g., `don t` → `don't`).
3. Expanded contractions (e.g., `can't` → `cannot`).
4. Removed URLs, emojis, special characters, @mentions, and hashtags.
5. Normalized possessives (`women's` → `women`) and common slang.
6. Split hashtags/mentions using `wordninja`.
7. Lowercased and cleaned tokens for modeling.

## 📄 Input

CSV file with `Text` and `Target 1` columns.

## 📤 Output

Same file with cleaned `Text` and preserved `Ori Text`.

## 🛠 Requirements

Install dependencies:
```bash
pip install pandas contractions wordninja


3. For the 'subtaskB' train and val sets, we need to filter the 'In Use' column during actual training, using only the entries where 'In Use' == 1. This is because, during data splitting for subtaskB, one domain is chosen as the test set, and the remaining seven domains may have targets that overlap with the test domain. These overlapping targets need to be masked. The 'in use' column is used for this masking.

4. In our paper, we proposed a method that uses MNLI pre-trained models for training. In this case, the noun phrases are prompted (which also affects 'mixed' because it also contains noun-phrase targets). Therefore, in the paths corresponding to 'noun phrase' and 'mixed', there is a 'prompt' folder that contains the data with the prompted noun phrase targets. Results based on NLI prompting (e.g., **BART-MNLI-ep**) were obtained using these data.
