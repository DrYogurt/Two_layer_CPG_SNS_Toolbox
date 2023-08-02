class Environment:
    def __init__(self,neural_model,bio_mechanical_model):
        self.neural_model = neural_model
        self.biomechanical_model = bio_mechanical_model
    def step(self):
        # Code to simulate a step in both the neural and biomechanical processes
        self.neural_model.step()
        self.biomechanical_model.step()

    def reset(self):
        # Code to reset both the neural and biomechanical models to their initial states
        self.neural_model.reset()
        self.biomechanical_model.reset()
