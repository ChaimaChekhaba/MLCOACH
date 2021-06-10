import math
import numpy
import pandas
import os
import glob

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib
from pickle import load

from Neo4jManager import Neo4jManager

class_metrics = {'class_name': str,
                 'class_complexity': float,
                 'coupling_between_object_classes': float,
                 'lack_of_cohesion_in_methods': float,
                 'npath_complexity': float,
                 'role': float}

pattern_metrics = {'number_of_model_classes': int,
                   'number_of_view_classes': int,
                   'number_of_controller_classes': int,
                   'number_of_presenter_classes': int,
                   'number_of_view_model_classes': int,
                   'model_class_complexity': float,
                   'model_coupling_between_object_classes': float,
                   'model_lack_of_cohesion_in_methods': float,
                   'model_npath_complexity': float,
                   'view_class_complexity': float,
                   'view_coupling_between_object_classes': float,
                   'view_lack_of_cohesion_in_methods': float,
                   'view_npath_complexity': float,
                   'controller_class_complexity': float,
                   'controller_coupling_between_object_classes': float,
                   'controller_lack_of_cohesion_in_methods': float,
                   'controller_npath_complexity': float,
                   'presenter_class_complexity': float,
                   'presenter_coupling_between_object_classes': float,
                   'presenter_lack_of_cohesion_in_methods': float,
                   'presenter_npath_complexity': float,
                   'view_model_class_complexity': float,
                   'view_model_coupling_between_object_classes': float,
                   'view_model_lack_of_cohesion_in_methods': float,
                   'view_model_npath_complexity': float}

# global parameters of the app
# the path of the apps
app_dir = "/home/chaima/Phd/JournalPaper/QualitativeStudy/app/"
tokens_path = '/home/chaima/Phd/JournalPaper/tokens.txt'
max_length = 9711


# replace the None values in all the data with the mean of the column
def replace_none_by_mean_all_data(df):
    # get the column names
    column_names = list(df)
    column_names.remove('app_key')
    if 'class_name' in column_names:
        column_names.remove('class_name')
        column_names.remove('TCOM_RAT')

    if 'Role' in column_names:
        column_names.remove('Role')
    if 'pattern' in column_names:
        column_names.remove('pattern')

    # get all the apps in the data
    app_keys = list(df['app_key'].unique())

    for app_key in app_keys:
        # replace the None value of each class in the app_key by the mean of the other classes without None
        df_app_key = df.loc[df['app_key'] == app_key]
        for column_name in column_names:
            df.loc[df['app_key'] == app_key, column_name] = df_app_key[column_name].replace('None', numpy.mean(
                pandas.to_numeric(df_app_key[column_name], errors='coerce')))

            # replace the None of apps where all the classes have a NaN by the mean of the column
            df[column_name] = df[column_name].replace(math.nan, numpy.mean(
                pandas.to_numeric(df[column_name], errors='coerce')))

    return df


def write(file, content):
    with open(file, "a+") as f:
        f.writelines(content)
        f.write("\n")


class Predictor:

    def __init__(self):
        self.model_predictor = "./model/cnn.h5"
        self.class_predictor = "./model/ClassModel.hdf5"
        self.Pattern_predictor = "./model/PatternClassifier.sav"
        self.class_scaler = "./model/class_scaler.pkl"
        self.pattern_scaler = "./model/pattern_scaler.pkl"
        self.result = None
        self.database = Neo4jManager()

    # compute code metrics of all the classes and the app
    def cleaning(self, app_key, oo_metrics):
        # the result file
        self.result = "./result/"+ app_key + ".txt"
        if os.path.exists(self.result):
            os.remove(self.result)
        # app_key: the name of the app in the database
        # oo_metrics: the file that contains the metrics computed with metricsReloaded
        # 1. get the model classes of the app using the learning-based approach
        path = app_dir + app_key + "/"
        model_classes = self.predict_model_classes(path)
        write(self.result, "Model classes (learning-based approach): ")
        for model in model_classes:
            write(self.result, str(model))
        write(self.result, "\n\n")
        # 2. get the model classes using the heurstic-based approach
        dataset, model_classes = self.database.readAppFromDatabase(app_key, oo_metrics)
        write(self.result, "Model classes (heuristic-based approach): ")
        for model in model_classes:
            write(self.result, str(model[0]))

        view_classes, controller_classes, presenter_classes, view_model_classes, none_classes = \
            self.predit_roles_of_classes(dataset)

        views = pandas.DataFrame(view_classes, columns=class_metrics)
        controllers = pandas.DataFrame(controller_classes, columns=class_metrics)
        presenters = pandas.DataFrame(presenter_classes, columns=class_metrics)
        view_models = pandas.DataFrame(view_model_classes, columns=class_metrics)
        class_metrics.pop('role', None)
        models = pandas.DataFrame(model_classes, columns=class_metrics)

        model = models[["class_complexity", "coupling_between_object_classes", "lack_of_cohesion_in_methods",
                        "npath_complexity"]].std(ddof=0).values.tolist()
        view = views[["class_complexity", "coupling_between_object_classes", "lack_of_cohesion_in_methods",
                      "npath_complexity"]].std(ddof=0).values.tolist()
        controller = controllers[["class_complexity", "coupling_between_object_classes", "lack_of_cohesion_in_methods",
                                  "npath_complexity"]].std(ddof=0).values.tolist()
        presenter = presenters[["class_complexity", "coupling_between_object_classes", "lack_of_cohesion_in_methods",
                                "npath_complexity"]].std(ddof=0).values.tolist()

        view_model = view_models[["class_complexity", "coupling_between_object_classes", "lack_of_cohesion_in_methods",
                                  "npath_complexity"]].std(ddof=0).values.tolist()

        pattern_feature_vector = [len(models), len(views), len(controllers), len(presenters), len(view_models)] + model \
                                 + view + controller + presenter + view_model
        class_metrics['role'] = str
        print("model classes", len(models))
        print("view classes", len(views))
        print("controller classes", len(controllers))
        print("presenter classes", len(presenters))
        print("view_model classes", len(view_models))

        self.predict_patterns_of_app(pattern_feature_vector)

    # predict the role of classes using ClassModel
    def predit_roles_of_classes(self, dataset):
        # dataset: the dataframe that contains all the feature vectors of classes
        dataset = replace_none_by_mean_all_data(dataset)
        metrics = dataset[["class_name", "class_complexity", "coupling_between_object_classes",
                           "lack_of_cohesion_in_methods", "npath_complexity"]]
        del dataset['class_name']
        del dataset['app_key']
        del dataset['TCOM_RAT']
        del dataset['class_complexity']
        del dataset['coupling_between_object_classes']
        del dataset['lack_of_cohesion_in_methods']
        del dataset['npath_complexity']

        nb_col = len(list(dataset)) - 1  # the number of columns in the dataframe
        # change pandas dataframe to numpy array
        X = dataset.iloc[:, :nb_col].values
        # check the infinite and Nan values
        X[X == numpy.inf] = 0
        X[X == numpy.nan] = 0
        X = numpy.array(X, dtype=float)
        # Normalize the data
        sc = load(open(self.class_scaler, 'rb'))
        X = sc.transform(X)
        model = load_model(self.class_predictor)
        yhat = model.predict(X)
        y = yhat.round()

        view_classes = list()
        controller_classes = list()
        presenter_classes = list()
        view_model_classes = list()
        none_classes = list()

        i = 0
        write(self.result, "\n\n")
        write(self.result, "Other classes: ")
        for index, class_ in metrics.iterrows():
            write(self.result, str(class_['class_name']) + str(y[i]))
            if y[i][0] == 1:
                view_classes.append([class_['class_name'], float(class_['class_complexity']),
                                     float(class_['coupling_between_object_classes']),
                                     float(class_['lack_of_cohesion_in_methods']),
                                     float(class_['npath_complexity']), yhat[i][0]])

            if y[i][1] == 1:
                controller_classes.append([class_['class_name'], float(class_['class_complexity']),
                                           float(class_['coupling_between_object_classes']),
                                           float(class_['lack_of_cohesion_in_methods']),
                                           float(class_['npath_complexity']), yhat[i][1]])

            if y[i][2] == 1:
                presenter_classes.append([class_['class_name'], float(class_['class_complexity']),
                                          float(class_['coupling_between_object_classes']),
                                          float(class_['lack_of_cohesion_in_methods']),
                                          float(class_['npath_complexity']), yhat[i][2]])

            if y[i][3] == 1:
                view_model_classes.append([class_['class_name'], float(class_['class_complexity']),
                                           float(class_['coupling_between_object_classes']),
                                           float(class_['lack_of_cohesion_in_methods']),
                                           float(class_['npath_complexity']), yhat[i][3]])

            if y[i][4] == 1:
                none_classes.append([class_['class_name'], float(class_['class_complexity']),
                                     float(class_['coupling_between_object_classes']),
                                     float(class_['lack_of_cohesion_in_methods']),
                                     float(class_['npath_complexity']), yhat[i][4]])
            i += 1
        print(len(view_classes))
        print(view_classes)
        return view_classes, controller_classes, presenter_classes, view_model_classes, none_classes

    # predict the patterns of the app using PatternModel
    def predict_patterns_of_app(self, pattern_feature_vector):
        pattern_feature_vector = [x if str(x) != 'nan' else -1 for x in pattern_feature_vector]
        dataset = pandas.DataFrame([pattern_feature_vector], columns=pattern_metrics)
        nb_col = len(list(dataset))  # the number of columns in the dataframe
        # change pandas dataframe to numpy array
        X = dataset.iloc[:, :nb_col].values
        # Normalize the data
        # Normalize the data
        sc = load(open(self.pattern_scaler, 'rb'))
        X[X == numpy.inf] = 0
        X[X == numpy.nan] = 0
        X = sc.transform(X)

        model = joblib.load(self.Pattern_predictor)
        # model = load_model(self.Pattern_predictor)
        yhat = model.predict(X)
        yprob = model.predict_proba(X)
        write(self.result, "\n\nPatterns applied in the app: \n" + str(yhat) + "\n" + str(yprob) + "\n")

    # predict the model classes in the appusing the cnn model
    def predict_model_classes(self, path):
        result = list()
        model = load_model(self.model_predictor)

        # for each java class in the app
        for class_file in glob.iglob(path + '**/*.java', recursive=True):
            class_doc = []
            with open(class_file, 'r') as f:
                class_doc.append(f.read())

            token = tokenize(class_doc)
            encoded_docs = token.texts_to_sequences(class_doc)
            padded_doc = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
            classes = model.predict(padded_doc)
            result.append([class_file, classes[0, 0]])

        return result


def tokenize(docs):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)
    return tokenizer
