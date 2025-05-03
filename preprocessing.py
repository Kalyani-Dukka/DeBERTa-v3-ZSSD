import pandas as pd
import re
import wordninja
from contractions import fix
from collections import defaultdict

# Define file path
file_path = r"C:\Users\Kalyani\Downloads\MINI\ezstance\subtaskA\mixed\prompt\raw_train_all_onecol.csv"

# Load dataset
df = pd.read_csv(file_path, encoding='utf-8')

# Initialize normalization dictionary
normalization_dict = defaultdict(str, {
    "we'll": "we will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'll": "they will",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "aren't": "are not",
    "i'm": "i am",
    "you've": "you have",
    "they've": "they have",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not"
})

def restore_apostrophes(text):
    """
    Fixes words like "don t" -> "don't", "can t" -> "can't", "women s" -> "women's".
    """
    text = re.sub(r"\b(don|can|won|he|she|it|we|they|you|that|who|what|let) t\b", r"\1't", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(i|we|you|they) ll\b", r"\1'll", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(i|we|you|they) ve\b", r"\1've", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(he|she|it|who) s\b", r"\1's", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(i) m\b", r"\1'm", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(women|teacher|child|doctor|student|president) s\b", r"\1's", text, flags=re.IGNORECASE)
    return text

def custom_fix_contractions(text):
    """Fix contractions properly after restoring apostrophes."""
    text = fix(text)  # Expand contractions

    replacements = {
        r"\bwon't\b": "will not",
        r"\bcan't\b": "cannot",
        r"\bshan't\b": "shall not",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
        r"\bit's\b": "it is",
        r"\bwe're\b": "we are",
        r"\bthey're\b": "they are",
        r"\byou're\b": "you are",
        r"\bwe'll\b": "we will",
        r"\bthey'll\b": "they will",
        r"\bwe've\b": "we have",
        r"\bthey've\b": "they have",
        r"\blet's\b": "let us",
        r"\bi'm\b": "i am",
        r"\byou've\b": "you have",
        r"\bshouldn't\b": "should not",
        r"\bcouldn't\b": "could not",
        r"\bwouldn't\b": "would not",
        r"\bdidn't\b": "did not"
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def preprocess_text(text, norm_dict):
    """Clean text by removing URLs, emojis, special characters, and normalizing tokens."""
    if not isinstance(text, str):  # Handle NaN or non-string values
        return ""
    
    # Restore apostrophes before contraction fixing
    text = restore_apostrophes(text)

    # Preserve common abbreviations like U.S., U.K., U.N., etc.
    text = re.sub(r"\b([A-Z])\.([A-Z])\.\b", r"\1\2", text)

    # Apply contraction fixes
    text = custom_fix_contractions(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Handle possessive apostrophes (e.g., women's → women, teacher's → teacher)
    text = re.sub(r"(\w+)'s", r"\1", text)

    # Remove special characters and emojis (basic handling)
    text = re.sub(r"[^\w\s#@.,!?]", "", text)

    # Remove unwanted hashtags
    text = re.sub(r"#SemST", "", text)

    # Normalize using dictionary
    tokens = text.split()
    cleaned_tokens = [norm_dict.get(token.lower(), token) for token in tokens]

    # Extract words, handling punctuation
    cleaned_tokens = re.findall(r"[A-Za-z0-9#@]+[,.]?|[,.!?&/\\<>=$]", " ".join(cleaned_tokens))
    cleaned_tokens = [[word.lower()] for word in cleaned_tokens]

    for i in range(len(cleaned_tokens)):
        word = cleaned_tokens[i][0].strip("#").strip("@")
        if word in norm_dict:
            cleaned_tokens[i] = norm_dict[word].split()
        elif cleaned_tokens[i][0].startswith("#") or cleaned_tokens[i][0].startswith("@"):  # Split hashtags & mentions
            cleaned_tokens[i] = wordninja.split(cleaned_tokens[i][0])

    return " ".join([word.lower() for sublist in cleaned_tokens for word in sublist])

# Apply text preprocessing
df['Text'] = df['Text'].astype(str).apply(lambda x: preprocess_text(x, normalization_dict))
df['Target 1'] = df['Target 1'].astype(str).apply(lambda x: preprocess_text(x, normalization_dict))

# Save the cleaned dataset
df.to_csv(file_path, index=False, encoding='utf-8')

print(f"Preprocessing complete. Cleaned data saved to {file_path}")
