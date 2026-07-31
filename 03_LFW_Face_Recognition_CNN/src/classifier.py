"""
LFW Face Recognition - SVM & k-NN Matching Classifier on Embeddings
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_face_classifier(X_train_embed, y_train, method='svm'):
    if method == 'svm':
        clf = SVC(kernel='rbf', C=10.0, probability=True, random_state=42)
    else:
        clf = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        
    clf.fit(X_train_embed, y_train)
    return clf


def evaluate_face_recognition(clf, X_test_embed, y_test, target_names):
    y_pred = clf.predict(X_test_embed)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    return acc, report
