import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from model import normalize_record

url = 'https://raw.githubusercontent.com/zm412/machine-learning-zoomcamp/refs/heads/main/midterm/automated_recruitment_analytics.csv'

df = pd.read_csv(url)

df.columns = df.columns.str.lower().str.replace(" ", "_")
df['outcome'] = (df['outcome'] == 'Hired').astype(int)

del df['candidate_id']
del df['resume_text']
del df['job_fit_score']
del df['job_id']

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

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)

X_val = dv.transform(val_dicts)

def research_best_model():
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    precisions = []
    recalls = []

    for t in thresholds:
        y_pred_bin = (y_pred >= t).astype(int)

        precisions.append(precision_score(y_val, y_pred_bin, zero_division=0))
        recalls.append(recall_score(y_val, y_pred_bin, zero_division=0))

        score = f1_score(y_val, y_pred_bin, zero_division=0)
        f1_scores.append(score)

    best_t_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_t_idx]
    best_f1 = f1_scores[best_t_idx]

    print(
        f'Optimal Threshold by F1-Score: {best_threshold:.2f} with F1:'
        f' {best_f1:.4f}'
    )

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', color='blue', linestyle='--')
    plt.plot(thresholds, recalls, label='Recall', color='green', linestyle='--')
    plt.plot(thresholds, f1_scores, label='F1-Score', color='red', linewidth=2)

    plt.axvline(x=best_threshold, color='black', linestyle=':', label=f'Best Threshold ({best_threshold:.2f})',)
    plt.title('Precision, Recall, F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def check_best_logistic_auc():
    scores = []

    for c_val in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        lr_model = LogisticRegression(
            C=c_val,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        )
        lr_model.fit(X_train, y_train)

        y_pred = lr_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((c_val, auc, y_pred))

    columns = ['C', 'auc', 'y_pred']
    df_scores = pd.DataFrame(scores, columns=columns)
    best_row = df_scores.loc[df_scores['auc'].idxmax()]

    return best_row


def check_best_decision_tree_auc():
    scores = []

    for depth in [4, 5, 6]:
        for s in [1, 5, 10, 15, 20, 500, 100, 200]:
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
            dt.fit(X_train, y_train)

            y_pred = dt.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            
            scores.append((depth, s, auc, y_pred))
    columns = ['max_depth', 'min_samples_leaf', 'auc', 'y_pred']
    df_scores = pd.DataFrame(scores, columns=columns)
    best_row = df_scores.loc[df_scores['auc'].idxmax()]

    return best_row

def check_best_forest_auc():
    scores = []

    for n in [10, 50, 100, 200]:
        for max_depth in range(10, 201, 10):
            rf = RandomForestClassifier(
              n_estimators=n,
              max_depth=max_depth,
              random_state=42,
              class_weight='balanced',
              n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)

            scores.append((n, max_depth, auc, y_pred))

    columns = ['n_estimators', 'max_depth', 'auc', 'y_pred']
    df_scores = pd.DataFrame(scores, columns=columns)
    best_row = df_scores.loc[df_scores['auc'].idxmax()]

    return best_row

def check_best_xgb_auc():
    scores = []

    for eta in [0.05, 0.1, 0.3]:
        for max_depth in [3, 5, 10]:
            for n_estimators in [50, 100, 200]:
                xgb_model = xgb.XGBClassifier(
                    learning_rate=eta,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1,
                )
                xgb_model.fit(X_train, y_train)

                y_pred = xgb_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)

                scores.append((eta, max_depth, n_estimators, auc, y_pred))

    columns = ['learning_rate', 'max_depth', 'n_estimators', 'auc', 'y_pred']
    df_scores = pd.DataFrame(scores, columns=columns)
    best_row = df_scores.loc[df_scores['auc'].idxmax()]

    return best_row

def get_comparison_result(best_threshold):
    lr = check_best_logistic_auc()
    dt = check_best_decision_tree_auc()
    rf = check_best_forest_auc()
    xgb = check_best_xgb_auc()

    comparison_df = pd.DataFrame({
        'act': y_val,
        'lr': (lr.y_pred >= best_threshold).astype(int),
        'dt': (dt.y_pred >= best_threshold).astype(int),
        'rf': (rf.y_pred >= best_threshold).astype(int),
        'xgb': (xgb.y_pred >= best_threshold).astype(int),
    })

    return comparison_df

def get_comparison_auc_result():
    lr = check_best_logistic_auc()
    dt = check_best_decision_tree_auc()
    rf = check_best_forest_auc()
    xgb = check_best_xgb_auc()

    comparison_df = {
        'lr': lr.auc,
        'dt': dt.auc,
        'rf': rf.auc,
        'xgb': xgb.auc,
    }

    return comparison_df

model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]

output_file = 'pipeline.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

