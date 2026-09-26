"""
What is Logistic Regression?
Logistic Regression is used when we want to predict a category.
"""
import numpy as np
x=np.array([[1],[2],[3],[5],[6],[8]])
y=np.array([0,0,0,1,1,1])
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y)
"""
5. Train the model
model.fit(X, y)
fit() means:
Learn from the given data.
The model looks at:
1 → Fail
2 → Fail
3 → Fail
5 → Pass
6 → Pass
8 → Pass
and learns the relationship.
"""
hours=float(input("Enter the number of hours(study): "))
prediction=model.predict([[hours]])
if prediction==0:
    print("Fail")
else:
    print("Pass")