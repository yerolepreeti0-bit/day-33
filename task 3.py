import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix

# Load dataset
data = pd.read_csv("Social_Network_Ads.csv")

print(data.head())
print(data.shape)

# Drop User ID (not useful for prediction)
data = data.drop('User ID',axis=1)

# Convert Gender to numbers
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Features and target
X = data[['Gender','Age','EstimatedSalary']]
y = data['Purchased']

# Split data
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.25,random_state=0
)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:",accuracy_score(y_test,y_pred))

# Confusion matrix
print(confusion_matrix(y_test,y_pred))