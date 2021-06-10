import pandas
from neo4jrestclient.client import GraphDatabase
from neo4jrestclient import client
import logging

# the list of OO metrics computed for every class in the entire dataset
oo_metrics = {
    "B": float,
    "CBO": float,
    "CLOC": float,
    "COM_RAT": float,
    "Command": float,
    "Cons": float,
    "CSA": float,
    "CSO": float,
    "CSOA": float,
    "Cyclic": float,
    "D": float,
    "Dcy": float,
    "Dcy1": float,
    "DIT": float,
    "Dpt": float,
    "Dpt1": float,
    "E": float,
    "Inner": float,
    "Inner1": float,
    "Jf": float,
    "JLOC": float,
    "Jm": float,
    "JTA": float,
    "JTM": float,
    "LCOM": float,
    "Level": float,
    "Level1": float,
    "LOC": float,
    "MPC": float,
    "N": float,
    "n": float,
    "NAAC": float,
    "NAIC": float,
    "NCLOC": float,
    "NOAC": float,
    "NOC": float,
    "NOIC": float,
    "NOOC": float,
    "NTP": float,
    "OCavg": float,
    "OCmax": float,
    "OPavg": float,
    "OSavg": float,
    "OSmax": float,
    "PDcy": float,
    "PDpt": float,
    "Query": float,
    "RFC": float,
    "STAT": float,
    "SUB": float,
    "TCOM_RAT": float,
    "TODO": float,
    "V": float,
    "WMC": float,
    "class_complexity": float,
    "coupling_between_object_classes": float,
    "lack_of_cohesion_in_methods": float,
    "npath_complexity": float,
    "class_name": str,
    "app_key": str,
    "Role": str
}

# the list of source code metrics of all the classes
code_metrics = {
    "number_of_gui_components": int,
    "number_of_gui_statements": int,
    "number_of_input_events_listeners": int,
    "number_of_lifecycle_events": int,
    "number_of_non_lifecycle_events": int,
    "number_of_direct_manipulation_of_models": int,
    "number_of_indirect_manipulation_of_models": int,
    "handle_input_events_in_xml_file": int,
    "has_data_tag_in_layout_file": bool,
    "does_use_data_binding": bool,
    "is_view_model": bool,
    "is_presenter_form": bool
}

# the list of source code metrics of the pattern dataset (apps)
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
                   'view_model_npath_complexity': float,
                   'pattern': str}


class Neo4jManager:

    def __init__(self, url="http://localhost:7474", username="neo4j", password="chaima"):
        self.db = GraphDatabase(url=url, username=username, password=password)
        # classes dataset
        self.classDataset = list()
        # apps dataset
        self.appDataset = list()

    # select all apps from the database for the training and test
    def readDataset(self):
        logging.info("Reading the apps from the database")
        # read all apps from the database
        query = 'match (a:App) where (a.number_of_classes > 0) RETURN a AS App'
        result = self.db.query(query, params={}, returns=client.Node)

        for r in result:
            app_key = r[0]['app_key']
            logging.info("Reading the " + app_key + " app:")
            # update the missed attributs when analysing the app with Cumin
            self.checkView(app_key)
            self.checkMissedAttributs(app_key)
            # compute the User interaction metrics for each UIC class.
            self.computeCodeMetricsOfUserInteractionClasses(app_key, False)
            self.computeSourceCodeMetricsOfIntermediateClasses(app_key, False)

        logging.info("End reading apps")

        logging.info("Generating the dataset files")
        self.generateDataset()
        logging.info("Dataset generation: Done")

    # read a single app from the database - prediction
    def readAppFromDatabase(self, app_key, oo_metrics_file):
        self.update_with_oo_metrics(oo_metrics_file)
        query = 'match (a:App{app_key: {app_key}}) where (a.number_of_classes > 0) RETURN a AS App'
        result = self.db.query(query, params={"app_key": app_key}, returns=client.Node)

        for r in result:
            app_key = r[0]['app_key']
            # update the missed attributs when analysing the app with Cumin
            self.checkView(app_key)
            self.checkMissedAttributs(app_key)
            # compute the User interaction metrics for each UIC class.
            self.computeCodeMetricsOfUserInteractionClasses(app_key, True)
            self.computeSourceCodeMetricsOfIntermediateClasses(app_key, True)

        return self.generateDataset(), self.computeSourceCodeMetricsOfModelClasses(app_key)

    # update the database with the views which are not correclty identified by Paprika
    def checkView(self, app_key):
        views = ["android.view.SurfaceView", "android.opengl.GLSurfaceView", "android.view.View",
                 "android.widget.LinearLayout", "android.widget.ImageView",
                 "android.inputmethodservice.KeyboardView",
                 "android.widget.TableLayout", "android.widget.TableRow", "android.widget.TextView",
                 "android.widget.ExpandableListView", "android.widget.ListView", "android.widget.FrameLayout",
                 "android.widget.AdapterView", "android.webkit.WebView",
                 "androidx.appcompat.widget.AppCompatImageView",
                 "android.widget.AnalogClock", "android.widget.AbsListView", "android.widget.AbsoluteLayout",
                 "android.widget.AbsSeekBar", "android.widget.AbsSpinner", "android.widget.ActionMenuView",
                 "android.widget.ActionMenuView.LayoutParams", "android.widget.AdapterView",
                 "android.widget.AdapterViewAnimator", "android.widget.AdapterViewFlipper",
                 "android.widget.AlphabetIndexer", "android.widget.AnalogClock",
                 "android.widget.AutoCompleteTextView", "android.widget.Button",
                 "android.widget.CalendarView", "android.widget.CheckBox", "android.widget.CheckedTextView",
                 "android.widget.Chronometer", "android.widget.CompoundButton",
                 "android.widget.DatePicker", "android.widget.DialerFilter", "android.widget.DigitalClock",
                 "android.widget.EdgeEffect", "android.widget.EditText", "android.widget.ExpandableListView",
                 "android.widget.Filter", "android.widget.FrameLayout", "android.widget.Gallery",
                 "android.widget.Gallery.LayoutParams", "android.widget.GridLayout",
                 "android.widget.GridLayout.Alignment", "android.widget.GridLayout.LayoutParams",
                 "android.widget.GridLayout.Spec", "android.widget.GridView",
                 "android.widget.HorizontalScrollView", "android.widget.ImageButton",
                 "android.widget.ImageSwitcher",
                 "android.widget.ImageView", "android.widget.LinearLayout",
                 "android.widget.LinearLayout.LayoutParams",
                 "android.widget.ListPopupWindow", "android.widget.ListView",
                 "android.widget.ListView.FixedViewInfo",
                 "android.widget.Magnifier", "android.widget.Magnifier.Builder", "android.widget.MediaController",
                 "android.widget.MultiAutoCompleteTextView",
                 "android.widget.MultiAutoCompleteTextView.CommaTokenizer",
                 "android.widget.NumberPicker", "android.widget.OverScroller", "android.widget.PopupMenu",
                 "android.widget.PopupWindow", "android.widget.ProgressBar", "android.widget.QuickContactBadge",
                 "android.widget.RadioButton", "android.widget.RadioGroup",
                 "android.widget.RadioGroup.LayoutParams",
                 "android.widget.RatingBar", "android.widget.RelativeLayout",
                 "android.widget.RelativeLayout.LayoutParams", "android.widget.RemoteViews",
                 "android.widget.Scroller", "android.widget.ScrollView",
                 "android.widget.SearchView", "android.widget.SeekBar", "android.widget.ShareActionProvider",
                 "android.widget.SlidingDrawer", "android.widget.Space", "android.widget.Spinner",
                 "android.widget.StackView", "android.widget.Switch", "android.widget.TabHost",
                 "android.widget.TableLayout", "android.widget.TableRow", "android.widget.TabWidget",
                 "android.widget.TextClock", "android.widget.TextSwitcher", "android.widget.TextView",
                 "android.widget.TimePicker", "android.widget.Toast", "android.widget.ToggleButton",
                 "android.widget.Toolbar", "android.widget.TwoLineListItem", "android.widget.VideoView",
                 "android.widget.ViewAnimator", "android.widget.ViewFlipper", "android.widget.ViewSwitcher",
                 "android.widget.ZoomButton", "android.widget.ZoomButtonsController", "android.widget.ZoomControls",
                 "android.preference.Preference", "android.preference.ListPreference"]
        query = 'MATCH (c:Class{app_key: {app_key}}) ' \
                'WHERE c.parent_name in ' + str(views) + ' AND not exists(c.is_view) ' \
                                                         ' SET c.is_view=true ' \
                                                         ' RETURN c.name, c.parent_name'

        self.db.query(query,
                      params={"app_key": app_key},
                      returns=(str, str),
                      data_contents=True)

    # set the missed properties
    def checkMissedAttributs(self, app_key):
        # set is_activity property
        query = 'MATCH (c:Class{app_key: {app_key}}) ' \
                'where (c.name ends with "Activity" or c.parent_name ends with "Activity") and not exists(' \
                'c.is_activity) ' \
                'SET c.is_activity=true ' \
                'return c.name, c.is_activity'

        self.db.query(query,
                      params={"app_key": app_key},
                      returns=(str, str),
                      data_contents=True)

        # set is_fragment property
        query = 'MATCH (c:Class{app_key: {app_key}}) ' \
                'where (c.name ends with "Fragment" or c.parent_name ends with "Fragment") and not exists(' \
                'c.is_fragment) ' \
                'SET c.is_fragment=true ' \
                'return c.name, c.is_fragment'

        self.db.query(query,
                      params={"app_key": app_key},
                      returns=(str, str),
                      data_contents=True)

        # set is_view property
        query = 'MATCH (c:Class{app_key: {app_key}}) ' \
                'where (c.name ends with "View" or c.parent_name ends with "View") and not exists(c.is_view) ' \
                'SET c.is_view = true ' \
                'return c.name, c.is_view'

        self.db.query(query,
                      params={"app_key": app_key},
                      returns=(str, str),
                      data_contents=True)

    # update the pattern of the apps in the database
    def updatePatternsOfApps(self, patterns_file="data/patterns.csv"):
        patterns = pandas.read_csv(patterns_file)
        # get all the apps in the data
        app_keys = list(patterns['app_key'].unique())

        for app_key in app_keys:
            ptn = str(patterns.loc[patterns['app_key'] == app_key, 'pattern'].values)
            query = 'MATCH (a:App{app_key: {app_key}}) SET a.patterns = ' + ptn + ' return a'

            self.db.query(query,
                          params={"app_key": app_key},
                          data_contents=True,
                          returns=client.Node)

    # get all activities, fragments, dialog or view of an app
    def computeCodeMetricsOfUserInteractionClasses(self, app_key, predict):
        query = 'MATCH (c:Class) ' \
                'WHERE c.app_key = {app_key} ' \
                'AND (c.is_activity=true OR c.is_fragment=true OR c.is_dialog=true OR c.is_view=true) ' \
                'RETURN c'

        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)

        for record in result:
            if not predict:
                roles = readRolesOfClass(record[0]['name'])
            else:
                roles = "to_predict"

            if 'is_view_model' in record[0].properties:
                is_view_model = 1
            else:
                is_view_model = 0

            row = [record[0]['number_of_gui_components'],
                   record[0]['number_of_gui_statements'],
                   record[0]['number_of_input_events_listeners'],
                   record[0]['number_of_lifecycle_events'],
                   record[0]['number_of_non_lifecycle_events'],
                   record[0]['handle_input_events_in_xml_file'],
                   record[0]['has_data_tag_in_layout_file'],
                   int(record[0]['does_use_data_binding']),
                   self.numberOfDirectManipulationOfModels(app_key, record[0]['name']),
                   self.numberOfInDirectManipulationOfModels(app_key, record[0]['name']),
                   is_view_model,
                   int(self.isPresenterForm(app_key, record[0]['name']))] \
                  + self.readOOMetrics(app_key, record[0]['name']) + \
                  [record[0]['name'], app_key, roles]

            self.classDataset.append(row)

    # get all classes that does not correspond to user interfaces elements
    def computeSourceCodeMetricsOfIntermediateClasses(self, app_key, predict):
        query = 'MATCH (c:Class)-[]->(v:Class)-[]->(m:Class) ' \
                'WHERE v.app_key = {app_key} ' \
                'AND NOT EXISTS(v.is_fragment) ' \
                'AND NOT EXISTS(v.is_activity) ' \
                'AND NOT EXISTS(v.is_view) ' \
                'AND NOT EXISTS(v.is_application) ' \
                'AND NOT EXISTS(v.is_broadcast_receiver) ' \
                'AND NOT EXISTS(v.is_content_provider) ' \
                'AND NOT EXISTS(v.is_dialog) ' \
                'AND NOT EXISTS(v.is_async_task) ' \
                'AND NOT EXISTS(v.is_interface) ' \
                'AND NOT EXISTS(v.is_model) ' \
                'RETURN DISTINCT v'

        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)

        for record in result:
            if not predict:
                roles = readRolesOfClass(record[0]['name'])
            else:
                roles = "to_predict"

            if 'is_view_model' in record[0].properties:
                is_view_model = 1
            else:
                is_view_model = 0

            row = [record[0]['number_of_gui_components'],
                   record[0]['number_of_gui_statements'],
                   record[0]['number_of_input_events_listeners'],
                   record[0]['number_of_lifecycle_events'],
                   record[0]['number_of_non_lifecycle_events'],
                   record[0]['handle_input_events_in_xml_file'],
                   record[0]['has_data_tag_in_layout_file'],
                   int(record[0]['does_use_data_binding']),
                   self.numberOfDirectManipulationOfModels(app_key, record[0]['name']),
                   self.numberOfInDirectManipulationOfModels(app_key, record[0]['name']),
                   is_view_model,
                   int(self.isPresenterForm(app_key, record[0]['name']))] \
                  + self.readOOMetrics(app_key, record[0]['name']) + \
                  [record[0]['name'], app_key, roles]
            self.classDataset.append(row)

    # get all model classes of the app
    def computeSourceCodeMetricsOfModelClasses(self, app_key):
        query = 'MATCH (v:Class) ' \
                'WHERE v.app_key = {app_key} ' \
                'AND v.is_model = true ' \
                'RETURN v'
        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)
        models = list()
        for record in result:
            models.append(
                [record[0]['name'], record[0]['class_complexity'], record[0]['coupling_between_object_classes'],
                 record[0]['lack_of_cohesion_in_methods'], record[0]['npath_complexity']])

        return models

    # compute metric direct manipulation of models
    def numberOfDirectManipulationOfModels(self, app_key, class_name):
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})-[:USES_CLASS|:COMPOSE]->(v:Class) ' \
                'WHERE v.is_model=true RETURN count(v)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)
        return result[0][0]

    # compute metric indirect manipulation of models
    def numberOfInDirectManipulationOfModels(self, app_key, class_name):
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})-[:USES_CLASS|:COMPOSE*2..5]->(v:Class {app_key: {' \
                'app_key}}) WHERE  v.is_model=true RETURN count(v)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)
        return result[0][0]

    # get the other metrics of the class
    def readOOMetrics(self, app_key, class_name):
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                'RETURN c.B, c.CBO, c.CLOC, c.COM_RAT,' \
                'c.Command, c.Cons, c.CSA, c.CSO, c.CSOA,' \
                'c.Cyclic, c.D, c.Dcy, c.Dcy1, c.DIT, c.Dpt,' \
                'c.Dpt1, c.E, c.Inner, c.Inner1, c.Jf, c.JLOC, c.Jm,' \
                'c.JTA, c.JTM, c.LCOM, c.Level, c.Level1, c.LOC,' \
                'c.MPC, c.N, c.n, c.NAAC, c.NAIC, c.NCLOC, c.NOAC,' \
                'c.NOC, c.NOIC, c.NOOC, c.NTP, c.OCavg, c.OCmax, ' \
                'c.OPavg, c.OSavg, c.OSmax, c.PDcy, c.PDpt, c.Query,' \
                'c.RFC, c.STAT, c.SUB, c.TCOM_RAT, c.TODO, c.V, c.WMC, c.class_complexity, ' \
                'c.coupling_between_object_classes, c.lack_of_cohesion_in_methods, c.npath_complexity'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=(str, str, str, str, str, str, str, str, str, str,
                                        str, str, str, str, str, str, str, str, str, str,
                                        str, str, str, str, str, str, str, str, str, str,
                                        str, str, str, str, str, str, str, str, str, str,
                                        str, str, str, str, str, str, str, str, str, str,
                                        str, str, str, str, str, str, str, str),
                               data_contents=True)
        return result[0]

    # check if the class could be a presenter
    def isPresenterForm(self, app_key, class_name):
        # case 1 : the presenter is abstract or an interface and the view uses an concrete child of the presenter
        query = 'MATCH (c:Class{app_key: {app_key}})' \
                '-[:COMPOSE]->(m:Class{app_key: {app_key}})<-[:EXTENDS|:IMPLEMENTS]-(cc:Class{name:{name}, app_key: {' \
                'app_key}})-[:USES_CLASS]->(c) WHERE (c.is_activity=true OR c.is_fragment=true OR c.is_dialog=true OR ' \
                'c.is_view=true) AND m.is_abstract=true OR m.is_interface=true ' \
                'RETURN count(cc)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)

        if result[0][0] > 0:
            return True
        else:
            # case 2: the view use the presenter and the presenter use directly the view
            query = 'MATCH (c:Class{app_key: {app_key}})' \
                    '-[:COMPOSE]->(cc:Class{name:{name}, app_key: {app_key}})-[:IMPLEMENTS|:EXTENDS]->(m:Class{' \
                    'app_key: {app_key}}) ' \
                    'WHERE (c.is_activity=true OR c.is_fragment=true OR c.is_dialog=true OR c.is_view=true) ' \
                    'RETURN count(cc)'

            result = self.db.query(query,
                                   params={"app_key": app_key, "name": class_name},
                                   returns=int,
                                   data_contents=True)

            if result[0][0] > 0:
                return True
            else:
                return False

    # generate dataset in csv format
    def generateDataset(self):
        code_metrics.update(oo_metrics)
        dataset = pandas.DataFrame(self.classDataset, columns=code_metrics)
        dataset.to_csv('classDataset.csv')
        exit()
        patterns = pandas.read_csv("data/patterns.csv")

        # generate the pattern dataset
        app_keys = list(dataset['app_key'].unique())

        for app_key in app_keys:
            df_app_key = dataset.loc[dataset['app_key'] == app_key]

            controller_classes = df_app_key.loc[df_app_key['Role'].str.contains("Controller"),
                                                ["class_complexity",
                                                 "coupling_between_object_classes",
                                                 "lack_of_cohesion_in_methods",
                                                 "npath_complexity"]]
            view_classes = df_app_key.loc[df_app_key['Role'].str.contains("View"),
                                          ["class_complexity",
                                           "coupling_between_object_classes",
                                           "lack_of_cohesion_in_methods",
                                           "npath_complexity"]]
            presenter_classes = df_app_key.loc[df_app_key['Role'].str.contains("Presenter"),
                                               ["class_complexity",
                                                "coupling_between_object_classes",
                                                "lack_of_cohesion_in_methods",
                                                "npath_complexity"]]
            model_classes = df_app_key.loc[df_app_key['Role'].str.contains("Model"),
                                           ["class_complexity",
                                            "coupling_between_object_classes",
                                            "lack_of_cohesion_in_methods",
                                            "npath_complexity"]]
            view_model_classes = df_app_key.loc[df_app_key['Role'].str.contains("ViewModel"),
                                                ["class_complexity",
                                                 "coupling_between_object_classes",
                                                 "lack_of_cohesion_in_methods",
                                                 "npath_complexity"]]

            number_of_model_classes = len(model_classes)
            number_of_view_classes = len(view_classes)
            number_of_controller_classes = len(controller_classes)
            number_of_presenter_classes = len(presenter_classes)
            number_of_view_model_classes = len(view_model_classes)

            model = model_classes.std(ddof=0).values.tolist()
            view = view_classes.std(ddof=0).values.tolist()
            controller = controller_classes.std(ddof=0).values.tolist()
            presenter = presenter_classes.std(ddof=0).values.tolist()
            view_model = view_model_classes.std(ddof=0).values.tolist()

            model = [x if str(x) != 'nan' else -1 for x in model]
            view = [x if str(x) != 'nan' else -1 for x in view]
            controller = [x if str(x) != 'nan' else -1 for x in controller]
            presenter = [x if str(x) != 'nan' else -1 for x in presenter]
            view_model = [x if str(x) != 'nan' else -1 for x in view_model]

            pattern_app = str(patterns.loc[patterns['app_key'] == app_key, ['pattern']].values)

            row = [number_of_model_classes, number_of_view_classes,
                   number_of_controller_classes, number_of_presenter_classes,
                   number_of_view_model_classes] + model + view + controller + presenter + view_model + [pattern_app]
            self.appDataset.append(row)
            print(app_key, len(row), row)
        patternDataset = pandas.DataFrame(self.appDataset, columns=pattern_metrics)
        patternDataset.to_csv('patternDataset.csv')
        print(patternDataset)

    # update the oo metrics of all apps in the dataset
    def update_with_oo_metrics(self, oo_metrics_file):
        import pandas
        import math
        data = pandas.read_csv(oo_metrics_file)
        metrics = list(data.columns)
        metrics.remove('App')
        metrics.remove('Class')

        for index, row in data.iterrows():
            app_key = row['App']
            class_name = row['Class']

            query = 'MATCH (c:Class{app_key: {app_key}, name: {class_name}}) SET '
            number_of_nan = data.iloc[index].isna().sum()
            if number_of_nan < 20:

                for metric in metrics:
                    if type(row[metric]) != str and not math.isnan(row[metric]):
                        query = query + 'c.' + metric + '=' + str(row[metric]) + ', '

                query = query[:-2] + ' RETURN c'
                # print(query)
                self.db.query(query,
                              params={"app_key": app_key, "class_name": class_name},
                              returns=client.Node,
                              data_contents=True)


# read the role(s) of the class from the annotation file
def readRolesOfClass(class_name):
    roles = pandas.read_csv('data/annotatedClasses.csv')
    applied_roles = roles.loc[roles['class_name'] == class_name]['role'].values
    if len(applied_roles) > 0:
        return list(applied_roles)[0]
    else:
        return 'Not annotated'
