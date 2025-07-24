import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
## edit for your .csv examples texts
df = pd.read_csv('emotions.csv')
df.columns = df.columns.str.strip()

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

## modify for your  dataset
model.fit(df['text'], df['label'])

joblib.dump(model, 'mood_model.pkl')
print("Model trained and saved as 'mood_model.pkl'")
