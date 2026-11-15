import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
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

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

auc_score = roc_auc_score(y_val, y_pred)
print(auc_score)
y_pred_bin = (y_pred >= 0.5).astype(int)


output_file = 'pipeline.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

thresholds = np.linspace(0, 1, 101)
f1_scores = []

for t in thresholds:
    y_pred_bin = (y_pred >= t).astype(int)

    score = f1_score(y_val, y_pred_bin, zero_division=0)
    f1_scores.append(score)

best_t_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_t_idx]
best_f1 = f1_scores[best_t_idx]

print(
    f'Optimal Threshold by F1-Score: {best_threshold:.2f} with F1:'
    f' {best_f1:.4f}'
)
def check_best_forest_attr():
    best_auc = 0
    best_params = {}

    for n in [10, 50, 100, 200]:
        for max_depth in [5, 10, 15, 20, None]:
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=max_depth,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            y_pred_rf = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_rf)

            if auc > best_auc:
                best_auc = auc
                best_params = {
                  'n_estimators': n,
                  'max_depth': max_depth,
                }

            print(
                f'n: {n}, max_depth: {max_depth} - AUC:'
                f' {auc:.4f}'
            )

            print(
                f'\nBest RF Params: {best_params} with Validation AUC: {best_auc:.4f}'
            )

forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    class_weight='balanced',
)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict_proba(X_val)[:, 1]
forest_auc = roc_auc_score(y_val, y_pred_forest)

comparison_df = pd.DataFrame(
    {
        'actual_outcome': y_val,
        'logistic': (y_pred >= 0.4).astype(int),
        'forest': ( y_pred_forest >= 0.4 ).astype(int),
        'y_pred': y_pred,
        'y_pred_forest': y_pred_forest,
        'outcome': y_val.mean(),
        'predicted_outcome': y_pred.mean()
    }
)

thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score), y_val.mean(), (y_pred >= t).mean())
    scores.append(score)


print(round(forest_auc, 3), round( auc_score, 3 ), 'AUC')

#comparison_df[comparison_df['actual_outcome']==1]
precisions = []
recalls = []
f1_list = []

for t in thresholds:
    y_pred_bin = (y_pred >= t).astype(int)
    precisions.append(
      precision_score(y_val, y_pred_bin, zero_division=0)
    )
    recalls.append(recall_score(y_val, y_pred_bin, zero_division=0))
    f1_list.append(f1_score(y_val, y_pred_bin, zero_division=0))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', color='blue', linestyle='--')
plt.plot(thresholds, recalls, label='Recall', color='green', linestyle='--')
plt.plot(thresholds, f1_list, label='F1-Score', color='red', linewidth=2)

plt.axvline(
    x=best_threshold,
    color='black',
    linestyle=':',
    label=f'Best Threshold ({best_threshold:.2f})',
)
plt.title('Precision, Recall, and F1-Score across Decision Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
