# %% [markdown]
"""
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
# %% [markdown]
"""
<!--*!{misc/title.html}-->
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import KMeansSMOTE
# %% [markdown]
"""
<!--*!{sections/q1.html}-->
"""
# %%
df = pd.read_csv('data/bank-additional-full (1).csv', sep = ';')

# %%
df = df.drop(
    [
        "default",
        "pdays",
        "previous",	
        "poutcome",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed"
    ], 
    axis = 1
)

# %%
df = pd.get_dummies(
    df, 
    columns = [
        "loan",
        "job",
        "marital",
        "housing",
        "contact",
        "day_of_week",
        "campaign",
        "month",
        "education"
    ],
    drop_first = True
)

# %%
y = pd.get_dummies(df["y"], drop_first = True)
X = df.drop(["y"], axis = 1)

# %%
def bar_plot(y):
    obs = len(y)
    plt.bar(
        ["No","Yes"],
        [len(y[y.yes==0])/obs, len(y[y.yes==1])/obs]
    )
    plt.ylabel("Percentage of Data")
    plt.show()

# %%
bar_plot(y)
# %%
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X.astype(int), y.astype(int), 
    test_size=0.3, 
    random_state=42
)
# %% [markdown]
"""
<!--*!{sections/q2.html}-->
"""
# %%
smote = KMeansSMOTE(
    random_state=42
)
# %%
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# %%
bar_plot(y_train_smote)
# %% [markdown]
"""
`KmeansSMOTE` is a method of oversampling that uses the KMeans algorithm to create synthetic data. It is a combination of `KMeans` and `SMOTE` algorithms. It was applied to the training data to balance the classes.
"""

# %% [markdown]
"""
<!--*!{sections/q3.html}-->
"""

# %%
dtree = DecisionTreeClassifier(max_depth = 3)
dtree.fit(X_train_smote, y_train_smote)

# %%
fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(4,4),
    dpi=300
)

plot_tree(
    dtree,
    filled=True,
    feature_names=X_train_smote.columns,
    class_names=["No","Yes"]
)

# %% [markdown]
"""
<!--*!{sections/q4.html}-->
"""
# %%
y_pred = dtree.predict(X_test)
y_true = y_test
cm_raw = confusion_matrix(y_true, y_pred)

# %%
class_labels = ['Negative', 'Positive']

# Plot the confusion matrix as a heatmap
sns.heatmap(
    cm_raw, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_labels, 
    yticklabels=class_labels
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# %% [markdown]
"""
<!--*!{sections/q5.html}-->
"""
# %%
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3),
    bootstrap_features=True,
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)
bag.fit(X_train_smote, y_train_smote)
# %%
confusion_matrix(y_test, bag.predict(X_test))
# %% [markdown]
"""
<!--*!{sections/q6.html}-->
"""
# %% 
boost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    random_state=42
)
boost.fit(X_train_smote, y_train_smote)
# %%
ConfusionMatrixDisplay(
    confusion_matrix(y_test, boost.predict(X_test))
).plot()

# %% [markdown]
"""
<!--*!{sections/q7.html}-->
"""

# %%
super = LogisticRegression()
# %%
def X_calc_super(X):
    return np.array(
        [
            bag.predict_proba(X)[:,1],
            boost.predict_proba(X)[:,1],
            dtree.predict_proba(X)[:,1]
        ]
    ).T
# %%
X_super = X_calc_super(X_train_smote)
# %%
super.fit(X_super, y_train_smote)
# %%
ConfusionMatrixDisplay(
    confusion_matrix(y_test, super.predict(X_calc_super(X_test)))
).plot()

# %%
temp = f"""
Coefficients:
- Bagging : {np.round(super.coef_[0][0], 2)}
- Boosting : {np.round(super.coef_[0][1], 2)}
- D. Tree: {np.round(super.coef_[0][2], 2)}
"""

print(temp)
# %% [markdown]
"""
In this simple exercise the best performance was achieved with the `AdaBoost` model. The `LogisticRegression` model was used as a meta learner to combine the predictions of the `bagging`, `boosting`, and `dtree` models. The coefficients of the `DecisionTree` model is negative, which means that it is the least important in the ensemble. The `Boosting` model has the highest coefficient, which means it is the most important in the ensemble.
"""