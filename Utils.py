from Neo4JDatabase import Neo4jManager


# reading and storing the annotation file in the database
def readAppDesign(filename):
    with open(filename) as file:
        design = file.readline()
        design = design.split("\n")[0]
        app_key = filename.split("\n")[0].split("/")[len(filename.split("/")) - 1].split(".txt")[0]
        print("app_key", app_key)
        db = Neo4jManager()
        db.updateDesignInApp(app_key, design)

        if design != 'NONE':
            line = file.readline()
            while line:
                if "Model:" in line:
                    classes = line.split("\n")[0].split("Model:")[1].split(", ")
                    role = "Model"

                if "View:" in line:
                    classes = line.split("\n")[0].split("View:")[1].split(", ")
                    role = "View"

                if "Controller:" in line:
                    classes = line.split("\n")[0].split("Controller:")[1].split(", ")
                    role = "Controller"

                if "Presenter:" in line:
                    classes = line.split("\n")[0].split("Presenter:")[1].split(", ")
                    role = "Presenter"

                if "ViewModel:" in line:
                    classes = line.split("\n")[0].split("ViewModel:")[1].split(", ")
                    role = "ViewModel"

                for cls in classes:
                    cls = cls.replace(" ", "")
                    db.updateRoleInClass(app_key, cls, role)
                    print(cls, role)

                line = file.readline()


def readFolder(path):
    import glob
    for filename in glob.glob(path + '/*.txt'):
        readAppDesign(filename)


# files = ["ZephyrLogger"]
# for f in files:
#    readAppDesign('/home/chaima/Phd/Mining Mobile Apps to recommend Refactorings/Mining mobile apps to discover '
#                   'design patterns and code smells/annotated/' + f + '.txt')
# readFolder("/home/chaima/Phd/Mining Mobile Apps to recommend Refactorings/Mining mobile apps to discover design
# patterns and code smells/annotated/")
