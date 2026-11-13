import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from model import normalize_record

url = 'automated_recruitment_analytics.csv'

df = pd.read_csv(url)

df.columns = df.columns.str.lower().str.replace(" ", "_")
df['outcome'] = (df['outcome'] == 'Hired').astype(int)

del df['candidate_id']
del df['resume_text']
del df['job_fit_score']
del df['job_id']


#    Index([['skills', 'education', 'experience_years', 'job_title', 'required_skills', 'outcome']],
#      dtype='str')
#

#print(df.isnull().sum())
#print(df['skills'].apply(lambda x: isinstance(x, str)).all())
#print(df['job_title'].apply(lambda x: isinstance(x, str)).all())
#print(df['experience_years'].apply(lambda x: isinstance(x, int)).all())
#print(df['required_skills'].apply(lambda x: isinstance(x, str)).all())
#print(df['outcome'].apply(lambda x: isinstance(x, str)).all())


unique_skill_strings = df['skills'].unique()

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['outcome'].values
y_val = df_val['outcome'].values
y_test = df_test['outcome'].values

del df_train['outcome']
del df_val['outcome']
del df_test['outcome']

train_dicts = normalize_record(df_train.to_dict(orient="records"))
val_dicts = normalize_record(df_val.to_dict(orient='records'))

del df['skills']
del df['required_skills']

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)

X_val = dv.transform(val_dicts)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

auc_score = roc_auc_score(y_val, y_pred)
print(auc_score)
y_pred_bin = (y_pred >= 0.5).astype(int)


output_file = 'pipeline.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

temp_data = {
    "job_title": "Web Developer",
    "experience_years": 10,
    "education": "M.Tech",
    "skills": "html, javascript, python, css, apis, git, React, SQL, java"
}

X_new = dv.transform([temp_data])

