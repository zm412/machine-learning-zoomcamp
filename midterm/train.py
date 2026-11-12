import pandas as pd
import numpy as np
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

def prepare_interaction_dicts(df):
    dicts = []
    for _, row in df.iterrows():
        job = str(row['job_title']).strip().lower().replace(' ', '_')
        
        row_dict = {
            'job_title': row['job_title'],
            'experience_years': row['experience_years'],
            'education': row['education']
        }

        candidate_skills = {s.strip().lower() for s in str(row['skills']).split(',') if s.strip()}
        required_for_row = {s.strip().lower() for s in str(row['required_skills']).split(',') if s.strip()}

        for req_skill in required_for_row:
            if req_skill in candidate_skills:
                feature_name = f"job__{job}__skill__{req_skill.replace(' ', '_')}"
                row_dict[feature_name] = 1

        dicts.append(row_dict)
    return dicts

#skills_set = set()
#
#for skills in unique_skill_strings:
#    for s in str(skills).split(','):
#        cleaned_skill = s.strip().lower()
#        if cleaned_skill:
#            skills_set.add(cleaned_skill)


#{'stakeholder management', 'aws', 'excel', 'accounting', 'css', 'python', 'kubernetes', 'documentation', 'project planning', 'incident response', 'git', 'algorithms', 'hr analytics', 'research', 'communication', 'html', 'power bi', 'agile', 'performance tuning', 'machine learning', 'react', 'teamwork', 'networking', 'problem solving', 'javascript', 'postgresql', 'siem', 'database design', 'apis', 'mlops', 'linux', 'statistics', 'sql', 'deep learning', 'docker', 'pytorch', 'financial modeling', 'testing', 'data visualization', 'budgeting', 'presentation', 'java', 'security auditing', 'time management', 'risk analysis', 'data structures', 'process mapping', 'risk management', 'backup and recovery', 'requirements analysis', 'leadership', 'pandas'}

# requirements {'stakeholder management', 'aws', 'excel', 'css', 'accounting', 'python', 'kubernetes', 'project planning', 'incident response', 'git', 'algorithms', 'hr analytics', 'communication', 'html', 'power bi', 'agile', 'performance tuning', 'machine learning', 'react', 'networking', 'javascript', 'postgresql', 'database design', 'apis', 'siem', 'mlops', 'linux', 'sql', 'statistics', 'deep learning', 'docker', 'pytorch', 'financial modeling', 'budgeting', 'data visualization', 'java', 'security auditing', 'data structures', 'risk analysis', 'process mapping', 'risk management', 'backup and recovery', 'requirements analysis', 'pandas'}


requirements_list = [
    'stakeholder management', 'aws', 'excel', 'css', 'accounting', 'python', 'kubernetes', 'project planning', 'incident response', 'git', 'algorithms', 'hr analytics', 'communication', 'html', 'power bi', 'agile', 'performance tuning', 'machine learning', 'react', 'networking', 'javascript', 'postgresql', 'database design', 'apis', 'siem', 'mlops', 'linux', 'sql', 'statistics', 'deep learning', 'docker', 'pytorch', 'financial modeling', 'budgeting', 'data visualization', 'java', 'security auditing', 'data structures', 'risk analysis', 'process mapping', 'risk management', 'backup and recovery', 'requirements analysis', 'pandas'
        ]

skills_list = [
    'stakeholder management', 
    'aws', 
    'excel', 
    'accounting', 
    'css', 
    'python', 
    'kubernetes', 
    'documentation', 
    'project planning', 
    'incident response', 
    'git', 
    'algorithms', 
    'hr analytics', 
    'research', 
    'communication', 
    'html', 
    'power bi', 
    'agile', 
    'performance tuning', 
    'machine learning', 
    'react', 
    'teamwork', 
    'networking', 
    'problem solving', 
    'javascript', 
    'postgresql', 
    'siem', 
    'database design', 
    'apis', 
    'mlops', 
    'linux', 
    'statistics', 
    'sql', 
    'deep learning', 
    'docker', 
    'pytorch', 
    'financial modeling', 
    'testing', 
    'data visualization', 
    'budgeting', 'presentation', 
    'java', 
    'security auditing', 
    'time management', 
    'risk analysis', 
    'data structures', 
    'process mapping', 
    'risk management', 
    'backup and recovery', 
    'requirements analysis', 
    'leadership', 
    'pandas'
    ]


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

def prepare_dicts(df):
    dicts = []
    for _, row in df.iterrows():
        job = str(row['job_title']).strip().lower().replace(' ', '_')
        
        row_dict = {
            'job_title': row['job_title'],
            'experience_years': row['experience_years'],
            'education': row['education']
        }

        candidate_skills = {s.strip().lower() for s in str(row['skills']).split(',') if s.strip()}
        required_for_row = {s.strip().lower() for s in str(row['required_skills']).split(',') if s.strip()}

        for req_skill in required_for_row:
            if req_skill in candidate_skills:
                feature_name = f"job__{job}__skill__{req_skill.replace(' ', '_')}"
                row_dict[feature_name] = 1

        dicts.append(row_dict)
    return dicts

del df['skills']
del df['required_skills']

train_dicts = prepare_dicts(df_train)
val_dicts = prepare_dicts(df_val)

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)

X_val = dv.transform(val_dicts)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

auc_score = roc_auc_score(y_val, y_pred)
print(auc_score)
print(pd.Series(y_train).value_counts(normalize=True))
y_pred_bin = (y_pred >= 0.5).astype(int)
print(classification_report(y_val, y_pred_bin))


