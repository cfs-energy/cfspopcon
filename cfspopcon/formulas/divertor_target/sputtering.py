from importlib.resources import as_file, files
import yaml

def read_sputtering_data():
    with as_file(files("cfspopcon.formulas.divertor_target").joinpath("sputtering_data_trim.yaml")) as filepath:
        with open(filepath) as f:
            data = yaml.safe_load(f)