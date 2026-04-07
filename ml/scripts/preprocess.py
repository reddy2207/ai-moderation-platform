import pandas as pd
import re

# ----------------------------
# LOAD DATASETS
# ----------------------------

jigsaw = pd.read_csv("../data/train.csv")
hate = pd.read_csv("../data/labeled_data.csv")

print("Datasets loaded ✅")

# ----------------------------
# PROCESS JIGSAW DATA
# ----------------------------

def map_jigsaw(row):
    if row[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].sum() == 0:
        return "safe"
    elif row['identity_hate'] == 1:
        return "hate"
    elif row['insult'] == 1:
        return "abuse"
    elif row['toxic'] == 1:
        return "toxic"
    else:
        return "other"

jigsaw['label'] = jigsaw.apply(map_jigsaw, axis=1)
jigsaw = jigsaw[['comment_text', 'label']]

# ----------------------------
# PROCESS HATE DATASET
# ----------------------------

def map_hate(label):
    if label == 0:
        return "hate"
    elif label == 1:
        return "abuse"
    else:
        return "safe"

hate['label'] = hate['class'].apply(map_hate)
hate = hate[['tweet', 'label']]
hate.columns = ['comment_text', 'label']

# ----------------------------
# COMBINE DATA
# ----------------------------

df = pd.concat([jigsaw, hate], ignore_index=True)

print("Datasets merged ✅")
print(df['label'].value_counts())

# ----------------------------
# CLEAN TEXT
# ----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['comment_text'].apply(clean_text)

print("Text cleaned ✅")

# ----------------------------
# SAVE FINAL DATASET
# ----------------------------

df[['clean_text', 'label']].to_csv("../data/final_dataset.csv", index=False)

print("Final dataset saved ✅")