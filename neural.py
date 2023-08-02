from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.neurons import  NonSpikingNeuronWithPersistentSodiumChannel
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
from sns_toolbox.connections import NonSpikingTransmissionSynapse
from sns_toolbox.networks import DifferentiatorNetwork


class NeuralModel:
    def __init__(self):
        # Initialization code for the neural model
        pass

    def step(self):
        # Code to simulate a step in the neural process
        print("NeuralModel step executed")

    def reset(self):
        # Code to reset the neural model to its initial state
        print("NeuralModel reset")
        
class SNSModel(NeuralModel):
    def __init__(self):
        # Initialization code for the neural model
        pass

    def step(self):
        # Code to simulate a step in the neural process
        print("SNSModel step executed")

    def reset(self):
        # Code to reset the neural model to its initial state
        print("SNSModel reset")
