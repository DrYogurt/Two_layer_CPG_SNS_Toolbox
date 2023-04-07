import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_RGs(data):
    """
    plot rhythm generator activity
    """
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    ax[0].plot(data['Time'], data['L_RG_Flx'],label='HC_ext',color='C0')
    ax[0].plot(data['Time'], data['L_RG_Ext'],label='HC_flx',color='C1')
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Potential (mV)')
    ax[0].set_title('Left Hindlimb Rhythm Generator')
    ax[0].legend()

    ax[1].plot(data['Time'], data['R_RG_Flx'],label='HC_ext',color='C0')
    ax[1].plot(data['Time'], data['R_RG_Ext'],label='HC_flx',color='C1')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Potential (mV)')
    ax[1].set_title('Right Hindlimb Rhythm Generator')
    ax[1].legend()
    plt.show()

def plot_PFs(data):
    """
    plot Pattern Formation activity for both the Hip and knee/ankle pf's
    """
    fig, ax = plt.subplots(2,2, figsize=(15,8))
    ax[0,0].plot(data['Time'], data['L_PF_Hip_Flx'],label='HC_ext',color='C0')
    ax[0,0].plot(data['Time'], data['L_PF_Hip_Ext'],label='HC_flx',color='C1')
    ax[0,0].set_xlabel('Time (ms)')
    ax[0,0].set_ylabel('Potential (mV)')
    ax[0,0].set_title('Left Hindlimb Hip PF CPG')
    ax[0,0].legend()

    ax[1,0].plot(data['Time'], data['L_PF_KA_Flx'],label='HC_ext',color='C0')
    ax[1,0].plot(data['Time'], data['L_PF_KA_Ext'],label='HC_flx',color='C1')
    ax[1,0].set_xlabel('Time (ms)')
    ax[1,0].set_ylabel('Potential (mV)')
    ax[1,0].set_title('Left Hindlimb Knee/Ankle PF CPG')
    ax[1,0].legend()

    ax[0,1].plot(data['Time'], data['R_PF_Hip_Flx'],label='HC_ext',color='C0')
    ax[0,1].plot(data['Time'], data['R_PF_Hip_Ext'],label='HC_flx',color='C1')
    ax[0,1].set_xlabel('Time (ms)')
    ax[0,1].set_ylabel('Potential (mV)')
    ax[0,1].set_title('Right Hindlimb Hip PF CPG')
    ax[0,1].legend()

    ax[1,1].plot(data['Time'], data['R_PF_KA_Flx'],label='HC_ext',color='C0')
    ax[1,1].plot(data['Time'], data['R_PF_KA_Ext'],label='HC_flx',color='C1')
    ax[1,1].set_xlabel('Time (ms)')
    ax[1,1].set_ylabel('Potential (mV)')
    ax[1,1].set_title('Right Hindlimb Knee/Ankle PF CPG')
    ax[1,1].legend()
    plt.show()

def plot_joint_activity(data):
    fig, ax = plt.subplots(1,3, figsize = (20,4))
    ax[0].plot(data['Time'], data['L_Hip_Joint_pos'],label='Left',color='C0')
    ax[0].plot(data['Time'], data['R_Hip_Joint_pos'],label='Right',color='C1', linestyle='--')
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Joint Angle (rad)')
    ax[0].set_title('Hip')
    ax[0].legend()

    ax[1].plot(data['Time'], data['L_Knee_Joint_pos'],label='Left',color='C0')
    ax[1].plot(data['Time'], data['R_Knee_Joint_pos'],label='Right',color='C1', linestyle='--')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Joint Angle (rad)')
    ax[1].set_title('Knee')
    ax[1].legend()

    ax[2].plot(data['Time'], data['L_Ankle_Joint_pos'],label='Left',color='C0')
    ax[2].plot(data['Time'], data['R_Ankle_Joint_pos'],label='Right',color='C1', linestyle='--')
    ax[2].set_xlabel('Time (ms)')
    ax[2].set_ylabel('Joint Angle (rad)')
    ax[2].set_title('Ankle')
    ax[2].legend()
    plt.show()

def plot_mn_activity(data):
    fig, ax = plt.subplots(3,2, figsize=(15, 10))
    ax[0,0].plot(data['Time'], data['L_MN_Hip_Ext'],label='Extensor',color='red')
    ax[0,0].plot(data['Time'], data['L_MN_Hip_Flx'],label='Flexor',color='green')
    ax[0,0].set_xlabel('Time (ms)')
    ax[0,0].set_ylabel('Potential (mV)')
    ax[0,0].set_title('Left Hip Motor Neuron Activity')
    ax[0,0].legend()

    ax[1,0].plot(data['Time'], data['L_MN_Knee_Ext'],label='Extensor',color='red')
    ax[1,0].plot(data['Time'], data['L_MN_Knee_Flx'],label='Flexor',color='green')
    ax[1,0].set_xlabel('Time (ms)')
    ax[1,0].set_ylabel('Potential (mV)')
    ax[1,0].set_title('Left Knee Motor Neuron Activity')
    ax[1,0].legend()

    ax[2,0].plot(data['Time'], data['L_MN_Ankle_Ext'],label='Extensor',color='red')
    ax[2,0].plot(data['Time'], data['L_MN_Ankle_Flx'],label='Flexor',color='green')
    ax[2,0].set_xlabel('Time (ms)')
    ax[2,0].set_ylabel('Potential (mV)')
    ax[2,0].set_title('Left Ankle Motor Neuron Activity')
    ax[2,0].legend()

    ax[0,1].plot(data['Time'], data['R_MN_Hip_Ext'],label='Extensor',color='red')
    ax[0,1].plot(data['Time'], data['R_MN_Hip_Flx'],label='Flexor',color='green')
    ax[0,1].set_xlabel('Time (ms)')
    ax[0,1].set_ylabel('Potential (mV)')
    ax[0,1].set_title('Right Hip Motor Neuron Activity')
    ax[0,1].legend()

    ax[1,1].plot(data['Time'], data['R_MN_Knee_Ext'],label='Extensor',color='red')
    ax[1,1].plot(data['Time'], data['R_MN_Knee_Flx'],label='Flexor',color='green')
    ax[1,1].set_xlabel('Time (ms)')
    ax[1,1].set_ylabel('Potential (mV)')
    ax[1,1].set_title('Right Knee Motor Neuron Activity')
    ax[1,1].legend()

    ax[2,1].plot(data['Time'], data['R_MN_Ankle_Ext'],label='Extensor',color='red')
    ax[2,1].plot(data['Time'], data['R_MN_Ankle_Flx'],label='Flexor',color='green')
    ax[2,1].set_xlabel('Time (ms)')
    ax[2,1].set_ylabel('Potential (mV)')
    ax[2,1].set_title('Ankle Ankle Motor Neuron Activity')
    ax[2,1].legend()
    plt.show()

def plot_sigmoid_muscle_activation():
    from sns_and_mujoco_sims import stim2tension
    potentials = np.arange(-110, -30, 0.1)
    activation = np.zeros(len(potentials))
    for i in range(len(potentials)):
        activation[i] = stim2tension(potentials[i])

    plt.plot(potentials, activation,color='blue', label='Sigmoid Clipped from 0-1')
    plt.ylabel('Muscle Activation')
    plt.xlabel('Neuron Potential (mV)')
    plt.title('Muscle Activation Curve')
    plt.text(-105, 0.7, r"$1/(1+e^{0.1532(-70-Potential)}) - 0.01$")
    plt.text(-101, 0.78, 'Clipped between 0 and 1')
    plt.show()

def plot_animatlab_joint_angles():

    animatlab_data = pd.read_csv('animatlab_L_joint_angles.txt', delimiter='\t')
    fig, ax = plt.subplots(1,3, figsize = (20,4))
    ax[0].plot(animatlab_data['Time']*1000, animatlab_data['LH_HipZ'],label='Animatlab',color='C0')
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Joint Angle (rad)')
    ax[0].set_title('Animatlab Simulation Hip')

    ax[1].plot(animatlab_data['Time']*1000, animatlab_data['LH_Knee'],label='Left',color='C0')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Joint Angle (rad)')
    ax[1].set_title('Animatlab Simulation Knee')

    ax[2].plot(animatlab_data['Time']*1000, animatlab_data['LH_AnkleZ'],label='Left',color='C0')
    ax[2].set_xlabel('Time (ms)')
    ax[2].set_ylabel('Joint Angle (rad)')
    ax[2].set_title('Animatlab Simulation Ankle')
    plt.show()

def sim_time_comparisons():

    time_comp_data = pd.read_csv('sim_time_comp.csv')

    print('Average run time in Animatlab: ', np.mean(time_comp_data['Animatlab Sim Times']))
    print('Average run time in SNS-Toolbox: ', np.mean(time_comp_data['SNS_Mujoco Sim Times']))

    plt.plot(time_comp_data['Animatlab Sim Times'], label='Animatlab')
    plt.plot(time_comp_data['SNS_Mujoco Sim Times'], label='SNS-Toolbox & Mujoco')
    plt.xlabel('Iteration')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()

def main():
    data = pd.read_csv('sim_outputs.csv')
    # plot_RGs(data)
    # plot_PFs(data)
    # plot_joint_activity(data)
    # plot_animatlab_joint_angles()
    # plot_mn_activity(data)
    # plot_sigmoid_muscle_activation()
    sim_time_comparisons()


if __name__ == '__main__':
    main()