import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier




# Load the data
train_data = pd.read_csv("C:\Users\hp\Downloads\SML Project\train.csv")
test_data = pd.read_csv("C:\Users\hp\Downloads\SML Project\test.csv")

# Drop ID column and separate features and target variable
X = train_data.drop(columns=["ID", "category"])
y = train_data["category"]
test_X = test_data.drop(columns=["ID"])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled = scaler.transform(test_X)

# Remove outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_scaled)
X_filtered = X_scaled[outliers != -1]
y_filtered = y[outliers != -1]

# Dimensionality reduction using TruncatedSVD
svd = TruncatedSVD(n_components=800, algorithm='arpack')
X_svd = svd.fit_transform(X_filtered)
test_X_svd = svd.transform(test_X_scaled)

# Clustering using KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_svd)
train_cluster_labels = kmeans.predict(X_svd)
test_cluster_labels = kmeans.predict(test_X_svd)

# Add cluster labels as additional features
X_svd_clusters = np.column_stack((X_svd, train_cluster_labels))
test_X_svd_clusters = np.column_stack((test_X_svd, test_cluster_labels))

# Train and evaluate the classification models using k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logreg = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
rf = RandomForestClassifier(n_estimators=100)

classifiers = [logreg, knn, xgb, rf]
classifier_names = ["Logistic Regression", "K-Nearest Neighbors", "XGBoost", "Random Forest"]

for clf, name in zip(classifiers, classifier_names):
    cv_scores = cross_val_score(clf, X_svd_clusters, y_filtered, cv=kfold, scoring='accuracy')
    print(f"{name} CV Accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
}

grid_search_xgb = GridSearchCV(xgb, param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_svd_clusters, y_filtered)
best_xgb = grid_search_xgb.best_estimator_

# Ensemble the classifiers using a Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', logreg), ('knn', knn), ('xgb', best_xgb), ('rf', rf)], voting='hard')
voting_clf.fit(X_svd_clusters, y_filtered)

# Make predictions on the test set
test_predictions = voting_clf.predict(test_X_svd_clusters)

# Prepare the submission file
submission = pd.DataFrame({"ID": test_data["ID"], "category": test_predictions})

# Save the submission file
submission.to_csv("submission.csv", index=False)

print("Submission file saved as 'submission.csv'")
