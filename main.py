from Prediction import Predictor
from Neo4jManager import Neo4jManager
import glob


def run():
    p = Predictor()
    for app in glob.iglob('./metricReloaded/*.csv', recursive=False):
        app_name = app.split('/')[-1].split('.')[0]
        print(app_name)
        p.cleaning(app_name, app)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Neo4jManager().readDataset()
