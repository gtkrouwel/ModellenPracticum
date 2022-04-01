# Abstract base class.
class Model:
    def get_name(self):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError
    
    def compute_t_cable(self, electricity_data, t_soil_data):
        raise NotImplementedError

class Naive(Model):
    def get_name(self):
        return "Naive"

    def __str__(self):
        return "T_cable(t) = T_soil(t)"  # TODO check if correct.

    def compute_t_cable(self, electricity_data, t_soil_data):
        pass # TODO

class Linear(Model):
    def get_name(self):
        return "Linear"

    def __str__(self):
        return "T_cable(t) = C * P(t) + T_soil(t)"  # TODO check if correct.

    def compute_t_cable(self, electricity_data, t_soil_data):
        pass # TODO

models = [Naive(), Linear()]

# You can do a for-loop over the models like so:
circuit_nos = []
for circuit_no in circuit_nos:
    electricity_data = []  # TODO load data with predefined function from preprocess.py.
    t_soil_data = []  # TODO import Davy's code and get soil temperature data.

    for model in models:
        t_cable_data = model.compute_t_cable(electricity_data, t_soil_data)
        # Do bayesian linear regression.
        print("Result for the", model.get_name(), "model with equation", model)
        # Show results.