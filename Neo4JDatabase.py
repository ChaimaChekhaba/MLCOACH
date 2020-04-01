# -*- coding: UTF-8 -*-
import json

from neo4jrestclient.client import GraphDatabase
from neo4jrestclient import client
from neo4jrestclient.exceptions import NotFoundError


class Neo4jManager:

    def __init__(self, url="http://localhost:7474", username="neo4j", password="chaima"):
        self.db = GraphDatabase(url=url, username=username, password=password)
        self.apps = dict()
        self.design = dict()
        print("starting reading the apps from database")
        self.getApps()
        print("reading apps done")

    # select all apps from the database
    def getApps(self):

        query = 'MATCH (a:App) WHERE (a.number_of_classes > 2 ) RETURN a AS App '
        result = self.db.query(query, returns=client.Node)
        for r in result:
            # self.apps[r[0]['app_key']] = [self.getAllModelsWithRoles(r[0]['app_key'])]
            self.apps[r[0]['app_key']] = [self.getAllModels(r[0]['app_key'])]
            self.apps[r[0]['app_key']].append(self.getAllDatabindingClasses(r[0]['app_key']))
            self.apps[r[0]['app_key']].append(self.getUserInterfaceClasses(r[0]['app_key']))
            self.apps[r[0]['app_key']].append(self.getAllClassesWithRoles(r[0]['app_key']))
            self.apps[r[0]['app_key']].append(self.gerAllClassesNotUI(r[0]['app_key']))
            self.design[r[0]['app_key']] = r[0]['design']
            print(r[0]['app_key'])
        self.toJson("apps.json")

    # select all classes of an app
    def getClasses(self, app_key):
        query = 'MATCH (c:Class) ' \
                'WHERE c.app_key = {app_key} ' \
                'RETURN c'
        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)

        app_classes = list()
        for record in result:
            app_classes.append(record[0]['name'])

        return app_classes

    # get all activities, fragments, dialog or view of an app
    def getUserInterfaceClasses(self, app_key):
        query = 'MATCH (c:Class) ' \
                'WHERE c.app_key = {app_key} ' \
                'AND (c.is_activity=true OR c.is_fragment=true OR c.is_dialog=true OR c.is_view=true) ' \
                'RETURN c'

        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)

        app_classes = dict()
        for record in result:
            app_classes[record[0]['name']] = [self.numberOfInputEventListener(app_key, record[0]['name']),
                                              self.numberOfLifeCycleEventHandler(app_key, record[0]['name']),
                                              self.numberOfNonLifeCycleEventHandler(app_key, record[0]['name']),
                                              self.numberOfDirectManipulationOfModels(app_key, record[0]['name']),
                                              self.numberOfInDirectManipulationOfModels(app_key, record[0]['name'])] \
                #  + self.getOtherMetrics(app_key, record[0]['name'])

        return app_classes

    # get all classes that doesn not correspond to user interfaces elements
    def gerAllClassesNotUI(self, app_key):
        models = self.apps[app_key][0]
        query = 'MATCH (c:Class)-[]->(v:Class)-[]->(m:Class) ' \
                'WHERE v.app_key = {app_key} ' \
                'AND NOT EXISTS(v.is_fragment) ' \
                'AND NOT EXISTS(v.is_activity) ' \
                'AND NOT EXISTS(v.is_view) ' \
                'AND NOT EXISTS(v.is_application) ' \
                'AND NOT EXISTS(v.is_broadcast_receiver) ' \
                'AND NOT EXISTS(v.is_content_provider) ' \
                'AND NOT EXISTS(v.is_static) ' \
                'AND NOT EXISTS(v.is_dialog) ' \
                'AND NOT EXISTS(v.is_asynck_task) ' \
                'AND NOT EXISTS(v.is_interface) ' \
                'RETURN v'
        # 'AND NOT EXISTS(v.is_service) ' \

        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)

        app_classes = dict()
        for record in result:
            app_classes[record[0]['name']] = [self.numberOfDirectManipulationOfModels(app_key, record[0]['name']),
                                              self.numberOfInDirectManipulationOfModels(app_key, record[0]['name']),
                                              int(self.isPresenterForm(app_key, record[0]['name']))] \
                                             + self.getOtherMetrics(app_key, record[0]['name'])
            # self.numberOfInputEventListener(app_key, record[0]['name']),
            # self.numberOfLifeCycleEventHandler(app_key, record[0]['name']),
            # self.numberOfNonLifeCycleEventHandler(app_key, record[0]['name']),
            # self.numberOfDirectManipulationOfModels(app_key, record[0]['name']),

        return app_classes

    # compute metric Input event Listener
    def numberOfInputEventListener(self, app_key, class_name):

        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                '-[r:INVOKE]->(v:Class) ' \
                'RETURN count(v)'
        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)

        return result[0][0]

    # compute metric life-cycle event handlers
    def numberOfLifeCycleEventHandler(self, app_key, class_name):

        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                '-[r:CLASS_OWNS_METHOD]->(v:Method) ' \
                'WHERE (v.name in ["onCreate", "onStart", "onRestart", ' \
                '"onResume", "onPause", "onStop", "onDestroy", ' \
                '"onAttach", "onCreateView", "onActivityCreated", ' \
                '"onDestroyView", "onDetach"]) OR ' \
                '(NOT EXISTS(v.is_static) AND v.modifier in ["public", "protected"] AND ' \
                'v.name STARTS WITH "On")  ' \
                'RETURN count(v)'
        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)

        return result[0][0]

    # compute metric non lifecycle event handler
    def numberOfNonLifeCycleEventHandler(self, app_key, class_name):

        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                '-[r:IMPLEMENTS]->(v:Class) ' \
                'WHERE (v.name in ["KeyEvent.Callback", ' \
                '"Application.ActivitylifecycleCallbacks", ' \
                '"Window.Callback", ' \
                '"LayoutInflater.Factory", ' \
                '"LayoutInflater.Factory2", ' \
                '"ComponentCallbacks", ' \
                '"ComponentCallbacks2", ' \
                '"View.OnCreateContextMenuListener"]) ' \
                'RETURN count(v) '
        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)

        return result[0][0]

    # get models of an app
    def getAllModels(self, app_key):
        query = 'MATCH (v:Class{app_key:{app_key}}) ' \
                'WHERE (v.role="Model") ' \
                'OR (NOT EXISTS(v.is_fragment) ' \
                'AND NOT EXISTS(v.is_activity) ' \
                'AND NOT EXISTS(v.is_view) ' \
                'AND NOT EXISTS(v.is_application) ' \
                'AND NOT EXISTS(v.is_dialog) ' \
                'AND NOT EXISTS(v.is_inner_class) ' \
                'AND NOT EXISTS(v.is_service) ' \
                'AND NOT EXISTS(v.is_broadcast_receiver) ' \
                'AND NOT EXISTS(v.is_content_provider) ' \
                'AND NOT EXISTS(v.is_asynck_task)) ' \
                'RETURN v'
        # 'AND NOT EXISTS(v.is_static) ' \

        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)

        models = list()
        for record in result:
            try:
                if record[0]['role'] == "Model":
                    models.append(record[0]['name'])

            except NotFoundError:
                if self.isModel(app_key, record[0]['name']):
                    models.append(record[0]['name'])

        return models

    # check if a class is a model or not
    def isModel(self, app_key, class_name):

        models = ["com.google.auto.value.AutoValue",
                  "com.litesuits.orm.db.annotation",
                  "com.google.gson.annotations",
                  "com.instagram.common.json.annotation",
                  "org.greenrobot.greendao.annotation",
                  "com.j256.ormlite",
                  "com.raizlabs.android.dbflow.annotation",
                  "com.orm.annotation",
                  "com.orm.annotation.SugarRecord",
                  "com.activeandroid.annotation",
                  "com.activeandroid.Model",
                  "nl.qbusict.cupboard.annotation",
                  "co.uk.rushorm.core.annotations",
                  "co.uk.rushorm.core.RushObject",
                  "co.uk.rushorm.core.Rush",
                  "io.requery",
                  "io.requery.Persistable",
                  "com.github.gfx.android.orma.annotation",
                  "io.realm.annotations",
                  "io.realm.RealmObject",
                  "io.realm.RealmModel",
                  "ollie.annotation",
                  "ollie.Model",
                  "org.orman.mapper.annotation",
                  "org.orman.mapper.Model",
                  "com.roscopeco.ormdroid",
                  "com.roscopeco.ormdroid.Entity",
                  "se.emilsjolander.sprinkles.annotations",
                  "se.emilsjolander.sprinkles.Model",
                  "com.annotatedsql.annotation",
                  "com.shizhefei.db.annotations",
                  "org.litepal.annotation",
                  "org.litepal.crud.DataSupport",
                  "com.yahoo.squidb.annotations",
                  "com.pushtorefresh.storio3.sqlite.annotations",
                  "com.github.dkharrat.nexusdata.core.ManagedObject",
                  "nl.elastique.poetry.json.annotations",
                  "net.simonvt.schematic.annotation",
                  "cn.ieclipse.aorm.annotation",
                  "android.arch.persistence.room",
                  "com.fasterxml.jackson.annotation",
                  "shillelagh",
                  "com.hendrix.triorm.annotations",
                  "java.io.Serializable",
                  "android.os.Parcel",
                  "android.os.Parcelable"]

        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                '-[r:USES_CLASS|CALL_CLASS]->(v:ExternalClass) ' \
                'WHERE (v.name in ' + str(models) + ') ' \
                                                    'RETURN count(v)'
        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)

        if result[0][0] > 0:
            return True
        # else:
        #    return self.isBeanModel(app_key, class_name)
        else:
            return False

    # check if the class is a bean model
    def isBeanModel(self, app_key, class_name):
        return self.hasFields(app_key, class_name) and \
               self.hasConstructors(app_key, class_name) and \
               self.isExtendingFrameworkClasses(app_key, class_name) and \
               self.hasSetters(app_key, class_name) and \
               self.hasGetters(app_key, class_name)

    def hasFields(self, app_key, class_name):
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                'RETURN c.number_of_attributes'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)
        if result[0][0] > 0:
            return True
        else:
            return False

    def hasConstructors(self, app_key, class_name):
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                '-[:CLASS_OWNS_METHOD]->(m:Method{is_init:true}) ' \
                'RETURN count(m)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)
        if result[0][0] > 0:
            return True
        else:
            return False

    def isExtendingFrameworkClasses(self, app_key, class_name):
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                '-[r:EXTENDS]->(v:ExternalClass) ' \
                'RETURN count(v) '

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)

        if result[0][0] > 0:
            return False
        else:
            query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                    'RETURN c.parent_name'

            result = self.db.query(query,
                                   params={"app_key": app_key, "name": class_name},
                                   returns=str,
                                   data_contents=True)

            return self.isParentFromFramework(app_key, result[0][0])

    def isParentFromFramework(self, app_key, parent_name):
        query = 'MATCH (c:ExternalClass{app_key: {app_key}}) ' \
                'WHERE {parent_name} CONTAINS c.name AND SIZE(c.name) > 0 ' \
                'RETURN count(c)'

        result = self.db.query(query,
                               params={"app_key": app_key, "parent_name": parent_name},
                               returns=int,
                               data_contents=True)

        if result[0][0] > 0:
            return False
        else:
            return True

    def hasGetters(self, app_key, class_name):
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                '-[:CLASS_OWNS_METHOD]->(m:Method{is_getter:true}) ' \
                'RETURN count(m)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)
        if result[0][0] > 0:
            return True
        else:
            return False

    def hasSetters(self, app_key, class_name):
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                '-[:CLASS_OWNS_METHOD]->(m:Method{is_setter:true}) ' \
                'RETURN count(m)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)
        if result[0][0] > 0:
            return True
        else:
            return False

    # check if the class could be a presenter
    def isPresenterForm(self, app_key, class_name):
        # case 1 : the presenter is abstract or an interface and the view uses an concrete child of the presenter
        query = 'MATCH (c:Class{app_key: {app_key}})' \
                '-[:COMPOSE]->(m:Class)<-[:EXTENDS|:IMPLEMENTS]-(cc:Class{name:{name}})-[:USES_CLASS]->(c) ' \
                'WHERE (c.is_activity=true OR c.is_fragment=true OR c.is_dialog=true OR c.is_view=true) ' \
                'AND m.is_abstract=true OR m.is_interface=true ' \
                'RETURN count(cc)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)

        if result[0][0] > 0:
            return True
        else:
            # case 2: the view use the presenter and the presenter use directly the
            query = 'MATCH (c:Class{app_key: {app_key}})' \
                    '-[:COMPOSE]->(cc:Class{name:{name}})-[:IMPLEMENTS|:EXTENDS]->(m:Class) ' \
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

    # compute metric direct manipulation of models
    def numberOfDirectManipulationOfModels(self, app_key, class_name):
        models = self.apps[app_key][0]
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})-[:USES_CLASS|:COMPOSE]->(v:Class) ' \
                'WHERE v.name in ' + str(models) + ' ' \
                                                   'RETURN count(v)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)
        return result[0][0]

    # compute metric indirect manipulation of models
    def numberOfInDirectManipulationOfModels(self, app_key, class_name):

        models = self.apps[app_key][0]
        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})-[:USES_CLASS|:COMPOSE*2..5]->(v:Class) ' \
                'WHERE v.name in ' + str(models) + ' ' \
                                                   'RETURN count(v)'

        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=int,
                               data_contents=True)
        return result[0][0]

    # get all data bindings classes
    def getAllDatabindingClasses(self, app_key):
        query = 'MATCH (c:Class{app_key: {app_key}})-[:EXTENDS|:REALIZE]->(v:ExternalClass) ' \
                'WHERE  v.name CONTAINS "Binding" ' \
                'RETURN c'

        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)
        dataBindingClasses = list()
        for record in result:
            # if self.isModel(app_key, record[0]['name']):
            dataBindingClasses.append(record[0]['name'])

        return dataBindingClasses

    # get all viewmodel class inherited from Android framework. it's not important now
    def getViewModelClasses(self, app_key):
        query = 'MATCH (c:Class{app_key: {app_key}})-[:EXTENDS|:REALIZE]->(v:ExternalClass) ' \
                'WHERE  v.name IN ["androidx.lifecycle.AndroidViewModel", "androidx.lifecycle.ViewModel"] ' \
                'RETURN c'

        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)
        viewModelClasses = list()
        for record in result:
            # if self.isModel(app_key, record[0]['name']):
            viewModelClasses.append(record[0]['name'])

        return viewModelClasses

    # get the other metrics of the class
    def getOtherMetrics(self, app_key, class_name):

        query = 'MATCH (c:Class{app_key: {app_key}, name:{name}})' \
                'RETURN c.npath_complexity, ' \
                'c.lack_of_cohesion_in_methods, ' \
                'c.number_of_methods, ' \
                'c.depth_of_inheritance, ' \
                'c.coupling_between_object_classes, ' \
                'c.class_complexity, ' \
                'c.number_of_attributes, ' \
                'c.number_of_implemented_interfaces'
        # 'c.number_of_children, ' \
        result = self.db.query(query,
                               params={"app_key": app_key, "name": class_name},
                               returns=(float, float, float, float, float, float, float, float),
                               data_contents=True)

        return result[0]

    # update the role of a class
    def updateRoleInClass(self, app_key, class_name, role):
        query = 'MATCH (c:Class{app_key: {app_key}})' \
                'WHERE  c.name ENDS WITH {class_name} ' \
                'SET c.role = {role} ' \
                'RETURN c'

        result = self.db.query(query,
                               params={"app_key": app_key, "class_name": class_name, "role": role},
                               returns=client.Node,
                               data_contents=True)

        if result:
            print("class role updated correctly ", app_key, class_name)
        else:
            print("class role update not performed ", app_key, class_name)

    # update the design of an app
    def updateDesignInApp(self, app_key, design):
        query = 'MATCH (c:App{app_key: {app_key}})' \
                'SET c.design = {design} ' \
                'RETURN c'

        result = self.db.query(query,
                               params={"app_key": app_key, "design": design},
                               returns=client.Node,
                               data_contents=True)
        if result:
            print("design updated correctly ", app_key)
        else:
            print("update not performed ", app_key)

    # get the models of an app
    def getAllModelsWithRoles(self, app_key):
        query = 'MATCH (v:Class) ' \
                'WHERE v.app_key = {app_key} ' \
                'AND v.role = "Model" ' \
                'AND NOT EXISTS(v.is_inner_class) ' \
                'RETURN v'
        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)

        models = list()
        for record in result:
            models.append(record[0]['name'])

        return models

    # get the the rest of classes with roles
    def getAllClassesWithRoles(self, app_key):
        query = 'MATCH (v:Class) ' \
                'WHERE v.app_key = {app_key} ' \
                'AND v.role in ["View", "Controller", "Presenter", "ViewModel"] ' \
                'RETURN v'
        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=client.Node,
                               data_contents=True)
        # 'AND NOT EXISTS(v.is_inner_class) ' \
        viewClasses = list()
        controllerClasses = list()
        presenterClasses = list()
        viewModelClasses = list()

        for record in result:

            if record[0]['role'] == 'View':
                viewClasses.append(record[0]['name'])

            if record[0]['role'] == 'ViewModel':
                viewModelClasses.append(record[0]['name'])

            if record[0]['role'] == "Controller":
                controllerClasses.append(record[0]['name'])

            if record[0]['role'] == "Presenter":
                presenterClasses.append(record[0]['name'])

        return viewClasses, controllerClasses, presenterClasses, viewModelClasses

    @staticmethod
    def write(file, content):
        with open(file, "a+") as f:
            f.writelines(content)
            f.write("\n\n\n")

    # export all the apps to json file
    def toJson(self, file):
        with open(file, "w") as f:
            json.dump(self.apps, f)
        with open('design_'+file, "w") as f:
            json.dump(self.design, f)

    # read apps from json file
    @staticmethod
    def readJson(file):
        with open(file) as json_file:
            data = json.load(json_file)

        return data

    def check_view(self, app_key):
        print(app_key)
        views = ["android.view.SurfaceView", "android.opengl.GLSurfaceView", "android.view.View",
                 "android.widget.LinearLayout", "android.widget.ImageView", "android.inputmethodservice.KeyboardView",
                 "android.widget.TableLayout", "android.widget.TableRow", "android.widget.TextView",
                 "android.widget.ExpandableListView", "android.widget.ListView", "android.widget.FrameLayout",
                 "android.widget.AdapterView", "android.webkit.WebView", "androidx.appcompat.widget.AppCompatImageView",
                 "android.widget.AnalogClock", "android.widget.AbsListView", "android.widget.AbsoluteLayout",
                 "android.widget.AbsSeekBar", "android.widget.AbsSpinner", "android.widget.ActionMenuView",
                 "android.widget.ActionMenuView.LayoutParams", "android.widget.AdapterView",
                 "android.widget.AdapterViewAnimator", "android.widget.AdapterViewFlipper",
                 "android.widget.AlphabetIndexer", "android.widget.AnalogClock",
                 "android.widget.ArrayAdapter", "android.widget.AutoCompleteTextView", "android.widget.BaseAdapter",
                 "android.widget.BaseExpandableListAdapter", "android.widget.Button", "android.widget.CalendarView",
                 "android.widget.CheckBox", "android.widget.CheckedTextView", "android.widget.Chronometer",
                 "android.widget.CompoundButton", "android.widget.CursorAdapter", "android.widget.CursorTreeAdapter",
                 "android.widget.DatePicker", "android.widget.DialerFilter", "android.widget.DigitalClock",
                 "android.widget.EdgeEffect", "android.widget.EditText", "android.widget.ExpandableListView",
                 "android.widget.Filter", "android.widget.FrameLayout", "android.widget.Gallery",
                 "android.widget.Gallery.LayoutParams", "android.widget.GridLayout",
                 "android.widget.GridLayout.Alignment", "android.widget.GridLayout.LayoutParams",
                 "android.widget.GridLayout.Spec", "android.widget.GridView", "android.widget.HeaderViewListAdapter",
                 "android.widget.HorizontalScrollView", "android.widget.ImageButton", "android.widget.ImageSwitcher",
                 "android.widget.ImageView", "android.widget.LinearLayout", "android.widget.LinearLayout.LayoutParams",
                 "android.widget.ListPopupWindow", "android.widget.ListView", "android.widget.ListView.FixedViewInfo",
                 "android.widget.Magnifier", "android.widget.Magnifier.Builder", "android.widget.MediaController",
                 "android.widget.MultiAutoCompleteTextView", "android.widget.MultiAutoCompleteTextView.CommaTokenizer",
                 "android.widget.NumberPicker", "android.widget.OverScroller", "android.widget.PopupMenu",
                 "android.widget.PopupWindow", "android.widget.ProgressBar", "android.widget.QuickContactBadge",
                 "android.widget.RadioButton", "android.widget.RadioGroup", "android.widget.RadioGroup.LayoutParams",
                 "android.widget.RatingBar", "android.widget.RelativeLayout",
                 "android.widget.RelativeLayout.LayoutParams", "android.widget.RemoteViews",
                 "android.widget.RemoteViewsService", "android.widget.ResourceCursorAdapter",
                 "android.widget.ResourceCursorTreeAdapter", "android.widget.Scroller", "android.widget.ScrollView",
                 "android.widget.SearchView", "android.widget.SeekBar", "android.widget.ShareActionProvider",
                 "android.widget.SimpleAdapter", "android.widget.SimpleCursorAdapter",
                 "android.widget.SimpleCursorTreeAdapter", "android.widget.SimpleExpandableListAdapter",
                 "android.widget.SlidingDrawer", "android.widget.Space", "android.widget.Spinner",
                 "android.widget.StackView", "android.widget.Switch", "android.widget.TabHost",
                 "android.widget.TableLayout", "android.widget.TableRow", "android.widget.TabWidget",
                 "android.widget.TextClock", "android.widget.TextSwitcher", "android.widget.TextView",
                 "android.widget.TimePicker", "android.widget.Toast", "android.widget.ToggleButton",
                 "android.widget.Toolbar", "android.widget.TwoLineListItem", "android.widget.VideoView",
                 "android.widget.ViewAnimator", "android.widget.ViewFlipper", "android.widget.ViewSwitcher",
                 "android.widget.ZoomButton", "android.widget.ZoomButtonsController", "android.widget.ZoomControls"]
        query = 'MATCH (c:Class{app_key: {app_key}}) ' \
                'WHERE c.parent_name in ' + str(views) + ' AND not exists(c.is_view) ' \
                                                         ' SET c.is_view=True ' \
                                                         ' RETURN c.name, c.parent_name'

        result = self.db.query(query,
                               params={"app_key": app_key},
                               returns=(str, str),
                               data_contents=True)

        for res in result:
            print(res)
