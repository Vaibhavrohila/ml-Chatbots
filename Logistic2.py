import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X=np.array([[1],[2],[3],[5],[6],[8]])
y=np.array([0,0,0,1,1,1])
#Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model=LogisticRegression()
model.fit(X_train, y_train)
#train only train data
prediction =model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print("Prediction:", prediction)
print("Actual:", y_test)
print("Accuracy:", accuracy)