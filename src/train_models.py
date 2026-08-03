from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier




def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression Model
    """

    # Create Model
    model = LogisticRegression(max_iter=1000)

    # Train Model
    model.fit(X_train, y_train)

    return model
def train_decision_tree(X_train, y_train):
    """
    Train Decision Tree Model
    """

    # Create Model
    model = DecisionTreeClassifier(
        random_state=42
    )

    # Train Model
    model.fit(X_train, y_train)

    return model

def train_knn(X_train, y_train):
    """
    Train KNN Model
    """

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train)

    return model

def train_naive_bayes(X_train, y_train):
    """
    Train Gaussian Naive Bayes Model
    """

    model = GaussianNB()

    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Model
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model