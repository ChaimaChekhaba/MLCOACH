import random

from Neo4JDatabase import Neo4jManager
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, \
    plot_confusion_matrix, f1_score, plot_roc_curve
import pandas
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# features for user interaction classes
features_ui = {"numberOfInputEventListener": int,
               "numberOfLifeCycleEventHandler": int,
               "numberOfNonLifeCycleEventHandler": int,
               "numberOfDirectManipulationOfModels": int,
               "numberOfInDirectManipulationOfModels": int,
               "Role": int}
# features for intermediate classes
features_ic = {"numberOfDirectManipulationOfModels": int,
               "numberOfInDirectManipulationOfModels": int,
               "npath_complexity": float,
               "lack_of_cohesion_in_methods": float,
               "number_of_methods": int,
               "depth_of_inheritance": float,
               "coupling_between_object_classes": float,
               "class_complexity": float,
               "number_of_attributes": int,
               "number_of_implemented_interfaces": int,
               "is_presenterForm": int,
               "Role": int}


def main():
    print("starting the main program")
    # db = Neo4jManager()
    # apps = db.apps
    # design = db.design
    design = readJson("data/design_apps.json")
    apps = readJson("data/apps.json")
    training, test = split_data(design)

    cleaner = Cleaner(apps, training, test)
    data_set_ui_training, data_set_ui_test, data_set_ic_training, data_set_ic_test = cleaner.exportDataToCsv()

    visualize_data_pca(True)
    visualize_data_pca(False)

    models = [tree.DecisionTreeClassifier(),
              RandomForestClassifier(),
              BaggingClassifier(),
              AdaBoostClassifier(),
              GaussianNB(),
              LogisticRegression(),
              SVC()]

    print("classification starts")

    for model in models:
        # classify(model, data_set_ui_training, data_set_ui_test, "UI")
        # classify(model, data_set_ic_training, data_set_ic_test, "IC")
        modelUI = train_model(data_set_ui_training, model)
        restitueAppDesign(test, apps, modelUI, True)
        modelIC = train_model(data_set_ic_training, model)
        restitueAppDesign(test, apps, modelIC, False)

    print("classification done")


# PCA data visualization
def visualize_data_pca(ui):
    if ui is True:
        # View Controller None
        targets = [0, 1, 4]
        colors = ['g', 'b', 'r']
        name = "_UI"
        features_ = features_ui
        data = pandas.concat(
            [pandas.read_csv('data/ui_training.csv'),
             pandas.read_csv('data/ui_test.csv')],
            ignore_index=True
        )
    else:
        # Controller Presenter ViewModel None
        targets = [1, 2, 3, 4]
        colors = ['g', 'b', 'r', 'm']
        name = "_IC"
        features_ = features_ic
        data = pandas.concat(
            [pandas.read_csv('data/ic_training.csv'),
             pandas.read_csv('data/ic_test.csv')],
            ignore_index=True
        )
    data = data[~data.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]
    # x = data.loc[:, features].values
    numpy.seterr(divide='ignore', invalid='ignore')
    data_ = data.reindex(columns=features_)
    x = data_.loc[:, features_].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    x = numpy.nan_to_num(x)
    principalComponents = pca.fit_transform(x)
    principalDf = pandas.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    finalDf = pandas.concat([principalDf, data[['Role']]], axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['Role'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                   , finalDf.loc[indicesToKeep, 'PC2']
                   , c=color
                   , s=10)
    ax.legend(targets)
    ax.grid()
    plt.savefig('result/PCA' + name + '.png')


# executing all classification algorithms with cross validation, the confusion matrix and ROC curves are stored
# in the result directories
def cross_validation_data(model, X, y, n_splits):
    # we start with the UI classifier
    kf = KFold(n_splits=n_splits, shuffle=True)

    class_names = list(set(y))

    for i, (train, test) in enumerate(kf.split(X, y)):
        x_train = X.loc[train, :]
        y_train = y.loc[train]
        model.fit(x_train, y_train)
        x_test = X.loc[test, :]
        y_test = y.loc[test]
        plot_confusion_matrix_(model, x_test, y_test, class_names, str(i))
        y_predict = model.predict(x_test)
        compute_confusion_matrix_and_classification_report(y_test, y_predict)


# plot the ROC curve and compute the area under the curve
def ROC_curve(y_test, y_predict, algo):
    y_predict = y_predict[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    roc_auc = roc_auc_score(y_test, y_predict)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='green', lw=lw, label=algo + ' (auc = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ' + algo)
    plt.legend(loc="lower right")
    plt.savefig("./roc_curve_" + algo + ".png")

    return roc_auc


# compute confusion matrix and classification report
def compute_confusion_matrix_and_classification_report(y_test, y_predict):
    conf_matrix = confusion_matrix(y_test, y_predict)
    classification_report_ = classification_report(y_test, y_predict)
    print(conf_matrix)
    print(classification_report_)
    return conf_matrix, classification_report_


# display the confusion matrix of the model
def plot_confusion_matrix_(classifier, X_test, y_test, class_names, string):
    algo = type(classifier).__name__
    disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues, display_labels=class_names)
    title = "Confusion matrix " + algo
    disp.ax_.set_title(title)
    plt.savefig("result/confusion_matrix_" + algo + string + ".png")


# apply the split as cross validation split
def split_data(design_apps):
    mvc, mvp, mvvm, none = list(), list(), list(), list()
    for app_key, design in design_apps.items():
        if design == 'MVC':
            mvc.append(app_key)
        elif design == 'MVP':
            mvp.append(app_key)
        elif design == 'MVVM':
            mvvm.append(app_key)
        elif design == 'NONE':
            none.append(app_key)
        else:
            print(app_key)

    leng = int(len(mvc) * 0.2)
    mvc_test = random.sample(mvc, leng)
    leng = int(len(mvp) * 0.2)
    mvp_test = random.sample(mvp, leng)
    leng = int(len(mvvm) * 0.2)
    mvvm_test = random.sample(mvvm, leng)
    leng = int(len(none) * 0.2)
    none_test = random.sample(none, leng)

    mvc = [x for x in mvc if x not in mvc_test]
    mvp = [x for x in mvp if x not in mvp_test]
    mvvm = [x for x in mvvm if x not in mvvm_test]
    none = [x for x in none if x not in none_test]

    training = mvc + mvp + mvvm + none
    test = mvc_test + mvp_test + mvvm_test + none_test

    return training, test


# apply the classification and compute the confusion matrix
def classify(model, training, test, type):
    X = training.drop('Role', axis=1)
    y = training['Role']
    X = X.reset_index()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    model = model.fit(X, y)
    class_names = list(set(y))

    X_test = test.drop('Role', axis=1)
    y_test = test['Role']
    X_test = X_test.reset_index()

    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    y_predict = model.predict(X_test)

    compute_confusion_matrix_and_classification_report(y_test, y_predict)

    plot_confusion_matrix_(model, X_test, y_test, class_names, type)


class Cleaner:
    def __init__(self, apps_list, training, test):
        self.apps = apps_list
        self.training = training
        self.test = test
        # dataset for UI classification
        self.data_ui_training = list()
        self.data_ui_test = list()
        # dataset for IC classification
        self.data_ic_training = list()
        self.data_ic_test = list()

    # transform the data to vectors that could be used by classifier
    def generateDataVectors(self):

        for app_key, elements in self.apps.items():
            modelClasses = elements[0]
            dataBindingClasses = elements[1]
            userInterfaceClasses = elements[2]
            viewClasses = elements[3][0]
            controllerClasses = elements[3][1]
            presenterClasses = elements[3][2]
            viewModelClasses = elements[3][3]
            intermediateClasses = elements[4]

            for class_name, metrics in userInterfaceClasses.items():
                vector = list(metrics)
                inserted = False
                if class_name in viewClasses:
                    vector.append(0)  # 0 view
                    inserted = True

                if class_name in controllerClasses:
                    vector.append(1)  # 1 controller
                    inserted = True

                if class_name in presenterClasses:
                    vector.append(2)  # 2 presenter
                    inserted = True

                if class_name in viewModelClasses:
                    vector.append(3)  # 3 viewModel

                    inserted = True
                if not inserted and class_name not in modelClasses:
                    vector.append(4)  # 4 for None
                    inserted = True

                if inserted:
                    if app_key in self.training:
                        self.data_ui_training.append(vector)
                    else:
                        self.data_ui_test.append(vector)

            for class_name, metrics in intermediateClasses.items():
                vector = list(metrics)
                inserted = False
                if class_name in controllerClasses:
                    vector.append(1)  # 1 controller
                    inserted = True

                if class_name in presenterClasses and not inserted:
                    vector.append(2)  # 2 presenter
                    inserted = True

                if class_name in viewModelClasses and not inserted:
                    vector.append(3)  # 3 viewModel
                    inserted = True

                if not inserted and class_name not in modelClasses and class_name not in viewClasses and not inserted:
                    vector.append(4)  # 4 for None
                    inserted = True

                if inserted:
                    if app_key in self.training:
                        self.data_ic_training.append(vector)
                    else:
                        self.data_ic_test.append(vector)

    # export the datsets to csv
    def exportDataToCsv(self):
        self.generateDataVectors()
        ui_training = pandas.DataFrame(self.data_ui_training, columns=features_ui)
        ui_training.to_csv(r'data/ui_training.csv')

        ui_test = pandas.DataFrame(self.data_ui_test, columns=features_ui)
        ui_test.to_csv(r'data/ui_test.csv')

        ic_training = pandas.DataFrame(self.data_ic_training, columns=features_ic)
        ic_training.to_csv(r'data/ic_training.csv')

        ic_test = pandas.DataFrame(self.data_ic_test, columns=features_ic)
        ic_test.to_csv(r'data/ic_test.csv')

        ic_training = removeNaNInfiniteValues(ic_training)
        ic_test = removeNaNInfiniteValues(ic_test)
        ui_test = removeNaNInfiniteValues(ui_test)
        ui_training = removeNaNInfiniteValues(ui_training)

        return ui_training, ui_test, ic_training, ic_test


def removeNaNInfiniteValues(dataFrame):
    data = dataFrame[~dataFrame.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]

    return data


def readJson(file):
    with open(file) as json_file:
        data = json.load(json_file)

    return data


def restitueAppDesign(test_apps, apps, model, ui):

    for app_key in test_apps:
        prediction = dict()
        app = apps[app_key]
        if ui is True:
            userInterfaceClasses = app[2]

            if "Role" in features_ui:
                del features_ui['Role']
            for class_name, metrics in userInterfaceClasses.items():
                vector = list(metrics)
                prediction[class_name] = test_model(vector, model, features_ui)
                print(app_key, class_name, prediction[class_name])
        else:
            intermediateClasses = app[4]
            if 'Role' in features_ic:
                del features_ic['Role']

            for class_name, metrics in intermediateClasses.items():
                vector = list(metrics)
                prediction[class_name] = test_model(vector, model, features_ic)
                print(app_key, class_name, prediction[class_name])


def train_model(training, model):
    X = training.drop('Role', axis=1)
    y = training['Role']
    X = X.reset_index()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    model = model.fit(X, y)

    return model


def test_model(test, model, features):
    X_test = pandas.DataFrame([test], columns=features)
    X_test = removeNaNInfiniteValues(X_test)
    X_test = X_test.reset_index()
    return model.predict(X_test)


main()
