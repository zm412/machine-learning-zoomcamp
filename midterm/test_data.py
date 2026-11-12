import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
url = 'automated_recruitment_analytics.csv'

df = pd.read_csv(url)

df.columns = df.columns.str.lower().str.replace(" ", "_")
df['outcome'] = (df['outcome'] == 'Hired').astype(int)

del df['candidate_id']
del df['resume_text']
del df['job_id']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set a very large virtual terminal width so pandas doesn't wrap lines early
pd.set_option('display.width', 1000)

# Prevent truncation of long text inside cells
pd.set_option('display.max_colwidth', None)

