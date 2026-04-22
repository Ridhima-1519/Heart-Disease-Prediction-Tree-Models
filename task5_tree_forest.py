import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Decision Tree ------------------
dt = DecisionTreeClassifier(max_depth=3)  # control overfitting
dt.fit(X_train, y_train)

# Predictions
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Visualize tree
plt.figure(figsize=(12,8))
plot_tree(dt, filled=True, feature_names=X.columns)
plt.title("Decision Tree")
plt.show()

# ------------------ Random Forest ------------------
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# ------------------ Feature Importance ------------------
importances = rf.feature_importances_

plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# ------------------ Cross Validation ------------------
cv_scores = cross_val_score(rf, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average CV score:", cv_scores.mean())