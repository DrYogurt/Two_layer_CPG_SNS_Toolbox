# Description of Files
The files in this repository contain simulations for biomechanical models of a rat hindlimb in Animatlab and Mujoco as well as Synthetic Nervous Systems (SNSs) to control these models in Animatlab2 and SNS-Toolbox. 

## Animatlab Simulation Files
1. animatlab_model.aproj
    - Animatlab file that contains the biomechanical and neural models from: https://pdxscholar.library.pdx.edu/mengin_fac/243/
    - Biomechanical model: 2 muscles for each joint modeled with a linear-Hill muscle model. Bodies are simplified to boxes for computational effiecience
    - Neural model: each hindlimb has Rhythm generator -> Pattern Formation Layers -> Motor Circuits 
        - The knee and ankle share a patter formation layer
2. animatlab_model_Standalone.asim
    - Allows the animatlab simulation to be run in python
3. run_animatlab.py
    - Runs the animatlab standalone sim file in a python script

## SNS - Mujoco Sim Files
1. rat_hindlimb_both_legs.xml
    - Biomechanical model to run in Mujoco
    - Similar to the Animatlab model. Bodies show the bones of the hindlimb rather than simplified geometries. Muscles are modeled with a Hill Muscle model (as opposed to the linear-Hill muscle model). 
    - The main difference in the muscle model is that animatlab assumes a linear force vs velocity relationship
2. rat_sim_plots.py
    - Generates plots of data from the simulations
    - loads in the data from the sns-mujoco simulation Then gives the option to plot:
        - Rhythm generator activity
        - Pattern formation layer activity
        - Motor Neuron Activity
        - Joint Motion from the sns-mujoco simulation
        - joint motion from the animatlab simulation
        - The muscle activation sigmoid
        - simulation times for 100 simulations in animatlab and sns-toolbox w/ mujoco
3. sns_and_mujoco_sims.py
    - build_net(dt):  Builds the SNS model given a time step in milliseconds. Returns a compiled 
    - build_mujoco_model(xml_path,  mujoco_dt): Loads in the Mujoco model with Mujoco python bindings given the file path and time step in seconds. Initiallizes the position of the model to a resting state. Returns model and data which are used to run the simulation
    - stim2activation(stim): A sigmoidal curve converting motor neuron activity to a muscle activation between 0 and 1
        - In this model, all muscles share the same activation curve
        - Calculated as in Fletcher Youngs Dissertation: https://etd.ohiolink.edu/apexprod/rws_olink/r/1501/10?clear=10&p10_accession_num=case1648154057237043
            - We are given the basic function for a sigmoid:
            
              $act = \frac{1}{1 + e^{s(x_{offset} - stim)}} + y_{offset}$
        
            - where s is the steepness of the sigmoid and stim is the motor neuron potential.
            - The Motor neurons range in activity from -100 to -40 mV. 
            - We first set the $y_{offset}$ to -1% of the max activation, or $y_{offset} = -0.01$
            - We set $x_{offset}$ to be the average of this range ($x_{offset} = -70  mV$)
            - We then say that at the lower range ($stim=-100 mV$), we want $act=0$, and at the higher range ($stim=-40 mV$) we want $act=0.98$
            - We then use that to solve for the steepness in the equation:
            
              $\frac{1}{1 + e^{-30s}} - \frac{1}{1 + e^{30s}} + 0.98 = 0$
            
            - Solving for the steepness, s, we find $s = 0.1532$
        - run_sims(sns_dt, cpg_inputs, xml_path, num_steps, time_vec)
    - run_sims(sns_dt, cpg_inputs, xml_path, num_steps, time_vec): runs the sns and mujoco simulations together and outputs to a csv file
        - sns_dt is used to build the sns model, then converted to seconds and used with xml path to build the mujoco model
        - cpg_inputs is used to initially excite the cpg's in the neural models to begin oscillations
        - num_steps specifies the number of steps in the simulations
        - time_vec passes an array of time and is only used to write to a csv file
        - we first initialize variables and set other variables which reference the indices used in the sns and mujoco models. 
        - then we initialize the sns_inputs variable with the cpg_inputs and the muscle tensions from the mujoco model (zero for the first time step) 
        - We then repeat the following for num_steps:
            1. Take one time step in the sns_model using sns_inputs as the inputs to the model
            2. Take the Motor neuron potentials from the previous time step, run them through stim2activation and input the respective muscle in mj_data
            3. Take one time step in the mujoco model
            4. update the sns_inputs variable with feedback from the mujoco model
            5. record joint activity
        - After running the simulations we save data to sim_outputs.csv. The data saved includes:
            - Motor neuron activity
            - Neural activity from all CPG half-center neurons
            - All joint activity


 ## Other Files

 Other files include a network diagram,geometry files for the mujoco model, the .csv file from the sns-mujoco simulation, a .txt file with the joint activity from the animatlab simulation, and a sim_time_comp.csv which is the time is took to run simulations in animatlab and sns and mujoco (100 simulation times saved). 

 Images of the two models are also available.

 Lasly the sns_mujoco_with_video.ipynb file is intended to run the sns-mujoco simulation and make a video of the rat, but this does not work properly at the time of this upload.

