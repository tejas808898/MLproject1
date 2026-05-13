# =========================================================
# LEGAL DOCUMENT CLASSIFICATION SYSTEM
# GOOGLE COLAB VERSION
# Upload Document and Classify
# =========================================================

# =========================================================
# STEP 1: Install Required Libraries
# =========================================================



# =========================================================
# STEP 2: Import Libraries
# =========================================================

import pandas as pd
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# =========================================================
# STEP 3: Upload Dataset CSV File
# =========================================================

print("UPLOAD legal_documents.csv FILE")

uploaded = files.upload()

# Read dataset
df = pd.read_csv("legal_documents.csv")

# Display dataset
print("\nDATASET LOADED SUCCESSFULLY\n")
print(df.head())

# =========================================================
# STEP 4: Prepare Data
# =========================================================

X = df["text"]
y = df["category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# STEP 5: Create Machine Learning Model
# =========================================================

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

# =========================================================
# STEP 6: Check Accuracy
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:", accuracy)

# =========================================================
# STEP 7: Upload Legal Document
# =========================================================

print("\nUPLOAD samples.txt FILE")

uploaded_doc = files.upload()

# Get uploaded filename
filename = list(uploaded_doc.keys())[0]

# Read uploaded text file
with open(filename, "r", encoding="utf-8") as file:
    document_text = file.read()

# Display uploaded document
print("\nDOCUMENT CONTENT:\n")
print(document_text)

# =========================================================
# STEP 8: Predict Category
# =========================================================

prediction = model.predict([document_text])

print("\n===================================")
print("PREDICTED CATEGORY:", prediction[0])
print("===================================")
