import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.neurons import  NonSpikingNeuronWithPersistentSodiumChannel
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
from sns_toolbox.connections import NonSpikingTransmissionSynapse
from sns_toolbox.networks import DifferentiatorNetwork

import numpy as np
import matplotlib.pyplot as plt
import time
import mujoco
import pandas as pd

def build_net(dt = 0.1):
    """
    See: https://pdxscholar.library.pdx.edu/mengin_fac/243/ for network details
    """

    # cpg parameters
    delta = 0.1
    Cm = 5
    Gm = 1
    Ena = 50
    Er = -60
    R = 20

    Sm = 0.2
    Sh = -0.6
    delEna = Ena
    Km = 1
    Kh = 0.5
    Em = -40
    Eh = -60
    delEm = Em
    delEh = Eh
    tauHmax = 350

    Gna = 1.5
    
    # reformat for sns-toolbox
    g_ion = [Gna]
    e_ion = [delEna]
    k_m = [Km]
    slope_m = [Sm]
    e_m = [delEm]
    k_h = [Kh]
    slope_h = [Sh]
    e_h = [delEh]
    tau_max_h = [tauHmax]
    
    # defining cpg neurons
    HC_neuron = NonSpikingNeuronWithPersistentSodiumChannel(membrane_capacitance=Cm, membrane_conductance=Gm,
                                                                g_ion=g_ion,e_ion=e_ion,
                                                                k_m=k_m,slope_m=slope_m,e_m=e_m,
                                                                k_h=k_h,slope_h=slope_h,e_h=e_h,tau_max_h=tau_max_h,
                                                                name='HC',color='orange', resting_potential=Er , bias = 1)
    
    cpg_interneuron = NonSpikingNeuron(membrane_capacitance=Cm, membrane_conductance=Gm, resting_potential=Er, name="IN", color='blue')

    # synaptic parameters for cpg
    gSyn = 2.749
    Esyn_In = -70
    Esyn_HC = -40
    gSyn_RG_to_PF = 0.1

    synapse_cpg = NonSpikingSynapse(max_conductance=gSyn, reversal_potential=Esyn_HC, e_hi = -25, e_lo = -60)
    synapse_IN = NonSpikingSynapse(max_conductance=gSyn, reversal_potential=Esyn_In, e_hi = -25, e_lo = -60)
    synapse_RG_to_PF = NonSpikingSynapse(max_conductance=gSyn_RG_to_PF, reversal_potential=Esyn_HC, e_hi = -40, e_lo = -60)
    
    # building the Rhythm generator cpg
    net = Network()
    # add neurons
    net.add_neuron(HC_neuron,name='RG_HC_ext',color='blue')
    net.add_neuron(HC_neuron,name='RG_HC_flx',color='orange')
    # add input to HC_ext to initialize
    #NOTE: the HC_ext for all cpgs will be the one which receives the initial excitement to drive motion
    net.add_input('RG_HC_ext')
    net.add_output('RG_HC_ext')
    net.add_output('RG_HC_flx')

    # adding the interneurons 
    net.add_neuron(cpg_interneuron,name='RG_IN_ext', color='gray')
    net.add_neuron(cpg_interneuron,name='RG_IN_flx', color='gray')

    # defining the synapses in the cpg
    net.add_connection(synapse_cpg,'RG_HC_ext','RG_IN_ext')
    net.add_connection(synapse_cpg,'RG_HC_flx','RG_IN_flx')
    net.add_connection(synapse_IN,'RG_IN_ext', 'RG_HC_flx')
    net.add_connection(synapse_IN,'RG_IN_flx', 'RG_HC_ext')


    #Now we build the two pattern formation cpgs
    # PF for the hip
    net.add_neuron(HC_neuron,name='PF_hip_HC_ext',color='blue')
    net.add_neuron(HC_neuron,name='PF_hip_HC_flx',color='orange')
    net.add_neuron(cpg_interneuron,name='PF_hip_IN_ext', color='gray')
    net.add_neuron(cpg_interneuron,name='PF_hip_IN_flx', color='gray')
    net.add_output('PF_hip_HC_ext')
    net.add_output('PF_hip_HC_flx')

    net.add_connection(synapse_cpg,'PF_hip_HC_ext','PF_hip_IN_ext')
    net.add_connection(synapse_cpg,'PF_hip_HC_flx','PF_hip_IN_flx')
    net.add_connection(synapse_IN,'PF_hip_IN_ext', 'PF_hip_HC_flx')
    net.add_connection(synapse_IN,'PF_hip_IN_flx', 'PF_hip_HC_ext')
    net.add_input('PF_hip_HC_ext')

    # PF for the ankle
    net.add_neuron(HC_neuron,name='PF_KA_HC_ext',color='blue')
    net.add_neuron(HC_neuron,name='PF_KA_HC_flx',color='orange')
    net.add_neuron(cpg_interneuron,name='PF_KA_IN_ext', color='gray')
    net.add_neuron(cpg_interneuron,name='PF_KA_IN_flx', color='gray')
    net.add_output('PF_KA_HC_ext')
    net.add_output('PF_KA_HC_flx')

    net.add_connection(synapse_cpg,'PF_KA_HC_ext','PF_KA_IN_ext')
    net.add_connection(synapse_cpg,'PF_KA_HC_flx','PF_KA_IN_flx')
    net.add_connection(synapse_IN,'PF_KA_IN_ext', 'PF_KA_HC_flx')
    net.add_connection(synapse_IN,'PF_KA_IN_flx', 'PF_KA_HC_ext')
    net.add_input('PF_KA_HC_ext')

    # connecting RG to PF layer
    net.add_connection(synapse_RG_to_PF,'RG_HC_ext','PF_hip_HC_ext')
    net.add_connection(synapse_RG_to_PF,'RG_HC_flx','PF_hip_HC_flx')
    net.add_connection(synapse_RG_to_PF,'RG_HC_ext','PF_KA_HC_ext')
    net.add_connection(synapse_RG_to_PF,'RG_HC_flx','PF_KA_HC_flx')

    #CMM networks for the hip, knee, and ankle
    Er_MN = -100
    # define neurons for the CMM network
    MN = NonSpikingNeuron(membrane_capacitance=Cm, membrane_conductance=Gm, resting_potential=Er_MN)
    basic_neuron = NonSpikingNeuron(membrane_capacitance=Cm, membrane_conductance=Gm, resting_potential=Er)

    # hip CMM
    net.add_neuron(MN, name="MN_hip_ext", color="red")
    net.add_neuron(MN, name="MN_hip_flx", color="green")
    net.add_neuron(basic_neuron, name="RC_hip_ext", color="red")
    net.add_neuron(basic_neuron, name="RC_hip_flx", color="green")
    net.add_neuron(basic_neuron, name="Ia_hip_ext", color="red")
    net.add_neuron(basic_neuron, name="Ia_hip_flx", color="green")
    # knee CMM
    net.add_neuron(MN, name="MN_knee_ext", color="red")
    net.add_neuron(MN, name="MN_knee_flx", color="green")
    net.add_neuron(basic_neuron, name="RC_knee_ext", color="red")
    net.add_neuron(basic_neuron, name="RC_knee_flx", color="green")
    net.add_neuron(basic_neuron, name="Ia_knee_ext", color="red")
    net.add_neuron(basic_neuron, name="Ia_knee_flx", color="green")
    # ankle CMM
    net.add_neuron(MN, name="MN_ankle_ext", color="red")
    net.add_neuron(MN, name="MN_ankle_flx", color="green")
    net.add_neuron(basic_neuron, name="RC_ankle_ext", color="red")
    net.add_neuron(basic_neuron, name="RC_ankle_flx", color="green")
    net.add_neuron(basic_neuron, name="Ia_ankle_ext", color="red")
    net.add_neuron(basic_neuron, name="Ia_ankle_flx", color="green")

    #adding outputs from the MN's
    net.add_output("MN_hip_ext")
    net.add_output("MN_hip_flx")
    net.add_output("MN_knee_ext")
    net.add_output("MN_knee_flx")
    net.add_output("MN_ankle_ext")
    net.add_output("MN_ankle_flx")

    #synapse definitions
    Esyn_PF_to_MN = -10
    e_hi_PF_to_MN = -50
    e_low_PF_to_MN = -60
    
    synapse_PF_to_MN_hip_ext = NonSpikingSynapse(max_conductance = 2.565, reversal_potential = Esyn_PF_to_MN, e_hi = e_hi_PF_to_MN, e_lo = e_low_PF_to_MN)
    synapse_PF_to_MN_hip_flx = NonSpikingSynapse(max_conductance = 3.632, reversal_potential = Esyn_PF_to_MN, e_hi = e_hi_PF_to_MN, e_lo = e_low_PF_to_MN)
    synapse_PF_to_MN_knee_ext = NonSpikingSynapse(max_conductance = 4.93, reversal_potential = Esyn_PF_to_MN, e_hi = e_hi_PF_to_MN, e_lo = e_low_PF_to_MN)
    synapse_PF_to_MN_knee_flx = NonSpikingSynapse(max_conductance = 1.516, reversal_potential = Esyn_PF_to_MN, e_hi = e_hi_PF_to_MN, e_lo = e_low_PF_to_MN)
    synapse_PF_to_MN_ankle_ext = NonSpikingSynapse(max_conductance = 4.054, reversal_potential = Esyn_PF_to_MN, e_hi = e_hi_PF_to_MN, e_lo = e_low_PF_to_MN)
    synapse_PF_to_MN_ankle_flx = NonSpikingSynapse(max_conductance = 4.522, reversal_potential = Esyn_PF_to_MN, e_hi = e_hi_PF_to_MN, e_lo = e_low_PF_to_MN)
    
    synapse_PF_to_Ia = NonSpikingSynapse(max_conductance = 0.5, reversal_potential = -40, e_hi = -40, e_lo = -60)
    synapse_Between_Ia = NonSpikingSynapse(max_conductance= 0.5, reversal_potential=-70, e_hi = -40, e_lo = -60)
    synapse_Ia_to_MN = NonSpikingSynapse(max_conductance = 2, reversal_potential = -100, e_hi = -40, e_lo = -60)
    synapse_MN_to_RC = NonSpikingSynapse(max_conductance = 0.5, reversal_potential = -40, e_hi = -10, e_lo = -100)
    synapse_Between_RC = NonSpikingSynapse(max_conductance = 0.5, reversal_potential = -70, e_hi = -40, e_lo = -60)
    synapse_RC_to_MN = NonSpikingSynapse(max_conductance = 0.5, reversal_potential = -100, e_hi = -40, e_lo = -60)
    synapse_RC_to_Ia = NonSpikingSynapse(max_conductance = 0.5, reversal_potential = -70, e_hi = -40, e_lo = -60)

    #synapses for the hip
    net.add_connection(synapse_PF_to_MN_hip_ext, 'PF_hip_HC_ext', 'MN_hip_ext')
    net.add_connection(synapse_PF_to_MN_hip_flx, 'PF_hip_HC_flx', 'MN_hip_flx')
    net.add_connection(synapse_PF_to_Ia, 'PF_hip_HC_ext', 'Ia_hip_ext')
    net.add_connection(synapse_PF_to_Ia, 'PF_hip_HC_flx', 'Ia_hip_flx')
    net.add_connection(synapse_Between_Ia, 'Ia_hip_ext', 'Ia_hip_flx')
    net.add_connection(synapse_Between_Ia, 'Ia_hip_flx', 'Ia_hip_ext')
    net.add_connection(synapse_Ia_to_MN, 'Ia_hip_ext', 'MN_hip_flx')
    net.add_connection(synapse_Ia_to_MN, 'Ia_hip_flx', 'MN_hip_ext')
    net.add_connection(synapse_MN_to_RC, 'MN_hip_ext', 'RC_hip_ext')
    net.add_connection(synapse_MN_to_RC, 'MN_hip_flx', 'RC_hip_flx')
    net.add_connection(synapse_Between_RC, 'RC_hip_ext', 'MN_hip_ext')
    net.add_connection(synapse_Between_RC, 'RC_hip_flx', 'MN_hip_flx')
    net.add_connection(synapse_RC_to_MN, 'RC_hip_ext', 'MN_hip_ext')
    net.add_connection(synapse_RC_to_MN, 'RC_hip_flx', 'MN_hip_flx')
    net.add_connection(synapse_RC_to_Ia, 'RC_hip_ext', 'Ia_hip_ext')
    net.add_connection(synapse_RC_to_Ia, 'RC_hip_ext', 'Ia_hip_flx')

    net.add_connection(synapse_PF_to_MN_knee_ext, 'PF_KA_HC_ext', 'MN_knee_ext')
    net.add_connection(synapse_PF_to_MN_knee_flx, 'PF_KA_HC_flx', 'MN_knee_flx')
    net.add_connection(synapse_PF_to_Ia, 'PF_KA_HC_ext', 'Ia_knee_ext')
    net.add_connection(synapse_PF_to_Ia, 'PF_KA_HC_flx', 'Ia_knee_flx')
    net.add_connection(synapse_Between_Ia, 'Ia_knee_ext', 'Ia_knee_flx')
    net.add_connection(synapse_Between_Ia, 'Ia_knee_flx', 'Ia_knee_ext')
    net.add_connection(synapse_Ia_to_MN, 'Ia_knee_ext', 'MN_knee_flx')
    net.add_connection(synapse_Ia_to_MN, 'Ia_knee_flx', 'MN_knee_ext')
    net.add_connection(synapse_MN_to_RC, 'MN_knee_ext', 'RC_knee_ext')
    net.add_connection(synapse_MN_to_RC, 'MN_knee_flx', 'RC_knee_flx')
    net.add_connection(synapse_Between_RC, 'RC_knee_ext', 'MN_knee_ext')
    net.add_connection(synapse_Between_RC, 'RC_knee_flx', 'MN_knee_flx')
    net.add_connection(synapse_RC_to_MN, 'RC_knee_ext', 'MN_knee_ext')
    net.add_connection(synapse_RC_to_MN, 'RC_knee_flx', 'MN_knee_flx')
    net.add_connection(synapse_RC_to_Ia, 'RC_knee_ext', 'Ia_knee_ext')
    net.add_connection(synapse_RC_to_Ia, 'RC_knee_ext', 'Ia_knee_flx')

    net.add_connection(synapse_PF_to_MN_ankle_ext, 'PF_KA_HC_ext', 'MN_ankle_ext')
    net.add_connection(synapse_PF_to_MN_ankle_flx, 'PF_KA_HC_flx', 'MN_ankle_flx')
    net.add_connection(synapse_PF_to_Ia, 'PF_KA_HC_ext', 'Ia_ankle_ext')
    net.add_connection(synapse_PF_to_Ia, 'PF_KA_HC_flx', 'Ia_ankle_flx')
    net.add_connection(synapse_Between_Ia, 'Ia_ankle_ext', 'Ia_ankle_flx')
    net.add_connection(synapse_Between_Ia, 'Ia_ankle_flx', 'Ia_ankle_ext')
    net.add_connection(synapse_Ia_to_MN, 'Ia_ankle_ext', 'MN_ankle_flx')
    net.add_connection(synapse_Ia_to_MN, 'Ia_ankle_flx', 'MN_ankle_ext')
    net.add_connection(synapse_MN_to_RC, 'MN_ankle_ext', 'RC_ankle_ext')
    net.add_connection(synapse_MN_to_RC, 'MN_ankle_flx', 'RC_ankle_flx')
    net.add_connection(synapse_Between_RC, 'RC_ankle_ext', 'MN_ankle_ext')
    net.add_connection(synapse_Between_RC, 'RC_ankle_flx', 'MN_ankle_flx')
    net.add_connection(synapse_RC_to_MN, 'RC_ankle_ext', 'MN_ankle_ext')
    net.add_connection(synapse_RC_to_MN, 'RC_ankle_flx', 'MN_ankle_flx')
    net.add_connection(synapse_RC_to_Ia, 'RC_ankle_ext', 'Ia_ankle_ext')
    net.add_connection(synapse_RC_to_Ia, 'RC_ankle_ext', 'Ia_ankle_flx')


    # adding neurons for feedback
    net.add_neuron(basic_neuron, name="Ia_IN_hip_ext", color="red")
    net.add_neuron(basic_neuron, name="Ia_IN_hip_flx", color="green")
    net.add_neuron(basic_neuron, name="Ib_IN_hip_ext", color="red")
    net.add_neuron(basic_neuron, name="Ib_IN_hip_flx", color="green")

    net.add_neuron(basic_neuron, name="Ia_IN_knee_ext", color="red")
    net.add_neuron(basic_neuron, name="Ia_IN_knee_flx", color="green")
    net.add_neuron(basic_neuron, name="Ib_IN_knee_ext", color="red")
    net.add_neuron(basic_neuron, name="Ib_IN_knee_flx", color="green")

    net.add_neuron(basic_neuron, name="Ia_IN_ankle_ext", color="red")
    net.add_neuron(basic_neuron, name="Ia_IN_ankle_flx", color="green")
    net.add_neuron(basic_neuron, name="Ib_IN_ankle_ext", color="red")
    net.add_neuron(basic_neuron, name="Ib_IN_ankle_flx", color="green")

    synapse_IaIN_Ia = NonSpikingSynapse(max_conductance = 0.5, reversal_potential = -40, e_hi = -40, e_lo = -60)
    synapse_IbIN_MN = NonSpikingSynapse(max_conductance = 0.59, reversal_potential = -10, e_hi = -40, e_lo = -60)
    synapse_PF_Ib = NonSpikingSynapse(max_conductance = 2.0, reversal_potential = -60, e_hi = -59, e_lo = -60)

    net.add_connection(synapse_IaIN_Ia,"Ia_IN_hip_ext","Ia_hip_ext")
    net.add_connection(synapse_IbIN_MN,"Ib_IN_hip_ext","MN_hip_ext")
    net.add_connection(synapse_PF_Ib,"PF_hip_HC_ext","Ib_IN_hip_ext")
    
    net.add_connection(synapse_IaIN_Ia,"Ia_IN_hip_flx","Ia_hip_flx")
    net.add_connection(synapse_IbIN_MN,"Ib_IN_hip_flx","MN_hip_flx")
    net.add_connection(synapse_PF_Ib,"PF_hip_HC_flx","Ib_IN_hip_flx")

    net.add_connection(synapse_IaIN_Ia,"Ia_IN_knee_ext","Ia_knee_ext")
    net.add_connection(synapse_IbIN_MN,"Ib_IN_knee_ext","MN_knee_ext")
    net.add_connection(synapse_PF_Ib,"PF_KA_HC_ext","Ib_IN_knee_ext")

    net.add_connection(synapse_IaIN_Ia,"Ia_IN_knee_flx","Ia_knee_flx")
    net.add_connection(synapse_IbIN_MN,"Ib_IN_knee_flx","MN_knee_flx")
    net.add_connection(synapse_PF_Ib,"PF_KA_HC_flx","Ib_IN_knee_flx")

    net.add_connection(synapse_IaIN_Ia,"Ia_IN_ankle_ext","Ia_ankle_ext")
    net.add_connection(synapse_IbIN_MN,"Ib_IN_ankle_ext","MN_ankle_ext")
    net.add_connection(synapse_PF_Ib,"PF_KA_HC_ext","Ib_IN_ankle_ext")

    net.add_connection(synapse_IaIN_Ia,"Ia_IN_ankle_flx","Ia_ankle_flx")
    net.add_connection(synapse_IbIN_MN,"Ib_IN_ankle_flx","MN_ankle_flx")
    net.add_connection(synapse_PF_Ib,"PF_KA_HC_flx","Ib_IN_ankle_flx")

    net.add_input("Ia_IN_hip_ext")
    net.add_input("Ia_IN_hip_flx")
    net.add_input("Ib_IN_hip_ext")
    net.add_input("Ib_IN_hip_flx")

    net.add_input("Ia_IN_knee_ext")
    net.add_input("Ia_IN_knee_flx")
    net.add_input("Ib_IN_knee_ext")
    net.add_input("Ib_IN_knee_flx")

    net.add_input("Ia_IN_ankle_ext")
    net.add_input("Ia_IN_ankle_flx")
    net.add_input("Ib_IN_ankle_ext")
    net.add_input("Ib_IN_ankle_flx")
    
    model = net.compile(backend='numpy',dt=dt)

    return model
        
def build_mujoco_model(xml_path = 'rat_hind_3_joint_free.xml',  mujoco_dt = 0.1/1000):
    """
    loads in the mujoco model to be used as the physics model
    :param xml_path: path to the xml mujoco model
    :return:
    """

    # load in the mujoco model and simulation
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos0 = np.array([0.0575931, -0.00159813, 7.01028e-18, 0.0, -0.000591287, -0.999527, 0.199988, -0.000591287, -0.999527, 0.199988]) 

    # setting the initial pose to be at the simulations resting point
    data.qpos = qpos0

    model.opt.timestep = mujoco_dt

    mujoco.mj_forward(model, data)

    return model, data

def stim2activation(stim):
    """
    converts from a neural potential to a muscle activation between 0 and 1 with a clipped sigmoid curve
    :param stim: MN potential in mV
    :return: act: muscle activation between 0 and 1
    """

    steepness = 0.1532
    x_off = -70
    y_offset = -0.01
    amp = 1
    act = amp/(1 + np.exp(steepness*(x_off-stim))) + y_offset
    act = np.clip(act, 0,1)
    return act

def run_sims(R_sns_model, L_sns_model, mj_model, mj_data, cpg_inputs, num_steps, time_vec, save_data=1):

    """
    runs the simulations and saves the data to sim_outputs.csv
    """

    R_hip_ext_ind = 0
    R_hip_flx_ind = 1
    R_knee_ext_ind = 2
    R_knee_flx_ind = 3
    R_ankle_ext_ind = 4
    R_ankle_flx_ind = 5
    L_hip_ext_ind = 6
    L_hip_flx_ind = 7
    L_knee_ext_ind = 8
    L_knee_flx_ind = 9
    L_ankle_ext_ind = 10
    L_ankle_flx_ind = 11

    R_hip_joint_ind = 7
    R_ankle_joint_ind = 8
    R_knee_joint_ind = 9
    L_hip_joint_ind = 4
    L_ankle_joint_ind = 5
    L_knee_joint_ind = 6


    
    R_hip_joint_pos = np.zeros(num_steps)
    R_knee_joint_pos = np.zeros(num_steps)
    R_ankle_joint_pos = np.zeros(num_steps)
    L_hip_joint_pos = np.zeros(num_steps)
    L_knee_joint_pos = np.zeros(num_steps)
    L_ankle_joint_pos = np.zeros(num_steps)
    
    R_sns_data = np.zeros([num_steps, 12])
    L_sns_data = np.zeros([num_steps, 12])

    sns_inputs_RH = np.concatenate([cpg_inputs[0,:], np.zeros(12)])
    sns_inputs_LH = np.concatenate([cpg_inputs[0,:], np.zeros(12)])
    
    for i in range(num_steps):
        # take a step in the sns_models
        L_sns_data[i,:] = L_sns_model(sns_inputs_LH)
        R_sns_data[i,:] = R_sns_model(sns_inputs_RH)
        # use the sns Motor neuron data to activate muscles
        mj_data.act[L_hip_ext_ind] = stim2activation(L_sns_data[i-1,6])
        mj_data.act[L_hip_flx_ind] = stim2activation(L_sns_data[i-1,7])
        mj_data.act[L_knee_ext_ind] = stim2activation(L_sns_data[i-1,8])
        mj_data.act[L_knee_flx_ind] = stim2activation(L_sns_data[i-1,9])
        mj_data.act[L_ankle_ext_ind] = stim2activation(L_sns_data[i-1,10])
        mj_data.act[L_ankle_flx_ind] = stim2activation(L_sns_data[i-1,11])

        mj_data.act[R_hip_ext_ind] = stim2activation(R_sns_data[i-1,6])
        mj_data.act[R_hip_flx_ind] = stim2activation(R_sns_data[i-1,7])
        mj_data.act[R_knee_ext_ind] = stim2activation(R_sns_data[i-1,8])
        mj_data.act[R_knee_flx_ind] = stim2activation(R_sns_data[i-1,9])
        mj_data.act[R_ankle_ext_ind] = stim2activation(R_sns_data[i-1,10])
        mj_data.act[R_ankle_flx_ind] = stim2activation(R_sns_data[i-1,11])

        # take one timestep in the mujoco model
        mujoco.mj_step(mj_model, mj_data)

        # record and scale the tension for feedback (the negative sign is included becuase muscle forces are reported as negative, this way we get a positive tension)
        R_muscle_tensions = - 0.00105*np.array([mj_data.actuator_force[R_hip_ext_ind],mj_data.actuator_force[R_hip_flx_ind],mj_data.actuator_force[R_hip_ext_ind],mj_data.actuator_force[R_hip_flx_ind],
                                              mj_data.actuator_force[R_knee_ext_ind],mj_data.actuator_force[R_knee_flx_ind],mj_data.actuator_force[R_knee_ext_ind],mj_data.actuator_force[R_knee_flx_ind],
                                              mj_data.actuator_force[R_ankle_ext_ind],mj_data.actuator_force[R_ankle_flx_ind],mj_data.actuator_force[R_ankle_ext_ind],mj_data.actuator_force[R_ankle_flx_ind]])
        
        L_muscle_tensions = - 0.00105*np.array([mj_data.actuator_force[L_hip_ext_ind],mj_data.actuator_force[L_hip_flx_ind],mj_data.actuator_force[L_hip_ext_ind],mj_data.actuator_force[L_hip_flx_ind],
                                              mj_data.actuator_force[L_knee_ext_ind],mj_data.actuator_force[L_knee_flx_ind],mj_data.actuator_force[L_knee_ext_ind],mj_data.actuator_force[L_knee_flx_ind],
                                              mj_data.actuator_force[L_ankle_ext_ind],mj_data.actuator_force[L_ankle_flx_ind],mj_data.actuator_force[L_ankle_ext_ind],mj_data.actuator_force[L_ankle_flx_ind]])
        
        sns_inputs_LH = np.concatenate([cpg_inputs[i,:],L_muscle_tensions])
        sns_inputs_RH = np.concatenate([cpg_inputs[i,:],R_muscle_tensions])

        L_hip_joint_pos[i] = mj_data.qpos[L_hip_joint_ind] 
        L_knee_joint_pos[i] = mj_data.qpos[L_knee_joint_ind] 
        L_ankle_joint_pos[i] = mj_data.qpos[L_ankle_joint_ind] 

        R_hip_joint_pos[i] = mj_data.qpos[R_hip_joint_ind] 
        R_knee_joint_pos[i] = mj_data.qpos[R_knee_joint_ind] 
        R_ankle_joint_pos[i] = mj_data.qpos[R_ankle_joint_ind] 

    

    if save_data:
        L_sns_data = L_sns_data.transpose()
        R_sns_data = R_sns_data.transpose()

        if os.path.isfile('sim_outputs.csv'):
            os.remove('sim_outputs.csv')

        L_RG_df = pd.DataFrame({'Time': time_vec})
        L_RG_df.to_csv('sim_outputs.csv', sep=',', index=False, header=True)

        csv_input = pd.read_csv('sim_outputs.csv')
        csv_input['L_RG_Ext'] = R_sns_data[:][0]
        csv_input['L_RG_Flx'] = L_sns_data[:][1]
        csv_input['L_PF_Hip_Ext'] = L_sns_data[:][2]
        csv_input['L_PF_Hip_Flx'] = L_sns_data[:][3]
        csv_input['L_PF_KA_Ext'] = L_sns_data[:][4]
        csv_input['L_PF_KA_Flx'] = L_sns_data[:][5]
        csv_input['L_MN_Hip_Ext'] = L_sns_data[:][6]
        csv_input['L_MN_Hip_Flx'] = L_sns_data[:][7]
        csv_input['L_MN_Knee_Ext'] = L_sns_data[:][8]
        csv_input['L_MN_Knee_Flx'] = L_sns_data[:][9]
        csv_input['L_MN_Ankle_Ext'] = L_sns_data[:][10]
        csv_input['L_MN_Ankle_Flx'] = L_sns_data[:][11]
        csv_input['L_Hip_Joint_pos'] =  L_hip_joint_pos
        csv_input['L_Knee_Joint_pos'] =  L_knee_joint_pos
        csv_input['L_Ankle_Joint_pos'] =  L_ankle_joint_pos

        csv_input['R_RG_Ext'] = R_sns_data[:][0]
        csv_input['R_RG_Flx'] = R_sns_data[:][1]
        csv_input['R_PF_Hip_Ext'] = R_sns_data[:][2]
        csv_input['R_PF_Hip_Flx'] = R_sns_data[:][3]
        csv_input['R_PF_KA_Ext'] = R_sns_data[:][4]
        csv_input['R_PF_KA_Flx'] = R_sns_data[:][5]
        csv_input['R_MN_Hip_Ext'] = R_sns_data[:][6]
        csv_input['R_MN_Hip_Flx'] = R_sns_data[:][7]
        csv_input['R_MN_Knee_Ext'] = R_sns_data[:][8]
        csv_input['R_MN_Knee_Flx'] = R_sns_data[:][9]
        csv_input['R_MN_Ankle_Ext'] = R_sns_data[:][10]
        csv_input['R_MN_Ankle_Flx'] = R_sns_data[:][11]
        csv_input['R_Hip_Joint_pos'] =  R_hip_joint_pos
        csv_input['R_Knee_Joint_pos'] =  R_knee_joint_pos
        csv_input['R_Ankle_Joint_pos'] =  R_ankle_joint_pos

        csv_input.to_csv('sim_outputs.csv', sep=',', index=False, header=True)


def main():

    #TODO: sim_time comparison
    #TODO: document process of running sns with mujoco, description of functions.

    xml_path = 'rat_hindlimb_both_legs.xml'   

    I = 0
    dt = .1 # in milliseconds
    tMax = 5000 # in milliseconds
    t = np.arange(0,tMax,dt)
    numSteps = np.size(t)
    Iapp = np.zeros([numSteps,3])
    # Iapp[tStart:tEnd] = I
    Ipert = np.zeros([numSteps,3])
    Ipert[1] = 1
    inputs = Iapp + Ipert

    R_sns_model = build_net(dt = dt)
    L_sns_model = build_net(dt = dt)
    mj_model, mj_data = build_mujoco_model(xml_path=xml_path, mujoco_dt = dt/1000)

    save_times = np.zeros(100)
    for i in range(100):
        t_start = time.time()
        run_sims(R_sns_model, L_sns_model, mj_model, mj_data, cpg_inputs = inputs, time_vec = t, num_steps = numSteps, save_data=0)
        save_times[i] = time.time() - t_start

    time_comp = pd.read_csv('sim_time_comp.csv', delimiter=',')
    time_comp['SNS_Mujoco Sim Times'] = save_times
    time_comp.to_csv('sim_time_comp.csv', sep=',', index=False, header=True)

    
    
if __name__ == '__main__':
    t = time.time()
    main()
    print('Time to complete: ', time.time() - t)