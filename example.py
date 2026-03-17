import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 1. Load the dataset
df = pd.read_csv(r'C:\Users\yerol\OneDrive\Desktop\Preeti-Intership\day 33\data.csv')
print(df.head())

# 2. Data Cleaning (Removing unnecessary columns)
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
print(df.head())

# 3. Train-Test Split
# Taking all columns except the first (diagnosis) as features (X)
# Taking the first column (diagnosis) as the target (y)
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:],
    df.iloc[:, 0],
    test_size=0.2,
    random_state=2
)

# 4. Feature Scaling (Crucial for KNN because it relies on distance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Initial Model Building (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 6. Evaluation
y_pred = knn.predict(X_test)
print(f"Accuracy with k=5: {accuracy_score(y_test, y_pred)}")

# 7. Finding the Optimal k
scores = []
for i in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# Display the scores for each k
for k, score in enumerate(scores, 1):
    print(f"k = {k}: {score:.4f}")

import matplotlib.pyplot as plt

plt.plot(range(1, 16), scores)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. k-value for KNN")
plt.show()

# 5. Initial Model Building (k=5)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Accuracy with k=5: {accuracy_score(y_test, y_pred)}")