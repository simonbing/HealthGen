"""
2021 Simon Bing, ETHZ, MPI IS

Application to load data and save it.
"""
from healthgen.apps.base_app import BaseApplication
from absl import flags, app

FLAGS = flags.FLAGS

class DataLoaderApplication(BaseApplication):
    def __init__(self):
        super().__init__()

    def run(self):
        X, y = self.data_loader.get_data()
        print('Loaded data!')

def main(argv):
    application = DataLoaderApplication()
    application.run()


if __name__ == '__main__':
    app.run(main)