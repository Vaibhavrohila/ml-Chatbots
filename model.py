import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([
    35,
    42,
    50,
    58,
    65,
    72,
    80,
    88
])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual:", y_test)
print("Predicted:", predictions)

error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)
new_hours = [[9]]

prediction = model.predict(new_hours)

print("Predicted marks:", prediction[0])