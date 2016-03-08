import numpy as np
import sklearn.svm as svm
import seaborn as sns
sns.set_style("whitegrid")

from sklearn import cross_validation as cv
from sklearn.datasets import load_iris
from sklearn.learning_curve import validation_curve, learning_curve

def plotValidationCurve(estimator, title, X, y, param_name, param_range, cv=5):
    trainScores, testScores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring="accuracy", )

    trainScoresMean = np.mean(trainScores, axis=1)
    trainScoresStd = np.std(trainScores, axis=1)
    testScoresMean = np.mean(testScores, axis=1)
    testScoresStd = np.std(testScores, axis=1)

    sns.plt.title(title)
    sns.plt.xlabel(param_name)
    sns.plt.ylabel("Accuracy Score")
    sns.plt.ylim(0.0, 1.1)
    sns.plt.semilogx(param_range, trainScoresMean, label="Training score", color="r")
    sns.plt.fill_between(param_range, trainScoresMean - trainScoresStd, trainScoresMean + trainScoresStd, alpha=0.2, color="r")
    sns.plt.semilogx(param_range, testScoresMean, label="Cross-validation score",color="b")
    sns.plt.fill_between(param_range, testScoresMean - testScoresStd, testScoresMean + testScoresStd, alpha=0.2, color="b")

    sns.plt.legend(loc="best")
    return sns.plt

def plotLearningCurve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    sns.plt.figure()
    sns.plt.title(title)
    if ylim is not None:
        sns.plt.ylim(*ylim)
    sns.plt.xlabel("Training examples")
    sns.plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    sns.plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
    sns.plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    sns.plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    sns.plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    sns.plt.legend(loc="best")
    return sns.plt


if __name__ == "__main__":
    iris = load_iris()

    X = np.array(iris.data)
    y = np.array(iris.target)

    print("Number of data points : ", len(y))
    print("No of features : ", X.shape[1])
    print("No of classes : ", len(set(y)))
    print("\nFeature Names : ", iris.feature_names)
    print("Class Names : ", iris.target_names, "\n")

    # Validation Curve C
    Cs = np.logspace(-2, 6, 10)
    title = "Validation Curve - Regularization Factor C"

    plot = plotValidationCurve(svm.SVC(random_state=0), title, X, y, param_name="C", param_range=Cs, cv=5)
    plot.show()

    # Validation Curve Gamma
    gammas = np.logspace(-6, 3, 10)
    title = "Validation Curve - Regularization Factor Gamma"

    plot = plotValidationCurve(svm.SVC(random_state=0), title, X, y, param_name="gamma", param_range=gammas, cv=5)
    plot.show()

    # Learning Curve
    crossValidation = cv.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)
    title = "Learning Curve - Support Vector Machine"

    plot = plotLearningCurve(svm.SVC(random_state=0), title, X, y, ylim=(0.0, 1.1), cv=crossValidation)
    plot.show()
