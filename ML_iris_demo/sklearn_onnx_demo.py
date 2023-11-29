from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from skl2onnx import to_onnx

import onnxruntime as rt

# import joblib
import numpy as np
iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size= 0.8, random_state=0)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

# Convert into ONNX format.

onx  = to_onnx(clf, X[:1])


with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Compute the prediction with onnxruntime.
import onnxruntime as rt

sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred_onx)
print("Accuracy:", accuracy)
