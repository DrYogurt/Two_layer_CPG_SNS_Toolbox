import os
os.environ['MKL_NUM_THREADS'] = "1"

import numpy as np
import pandas as pd
import subprocess as sub
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.signal
import scipy.stats
import scipy.interpolate

from typing import Dict
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import uuid



def execute_simulation(path_asim_file: Path, path_simulator: Path) -> None:  # sub.CompletedProcess:
    """
    exuceute_simulation runs a given .asim file through an Animatlab Simulator which is specified through
                        path_simulator. This path may differ on your computer.

    :param path_asim_file: file path to the .asim file
    :param path_simulator: file path to the .exe simulator. Simulator should be AnimatSimulator.exe
    :return: returns a completed process
    """
    executable = [str(path_simulator), str(path_asim_file)]
    # results = sub.run(executable, stdout=sub.DEVNULL, stderr=sub.DEVNULL, timeout=15)
    proc_status = sub.Popen(executable, stdout=sub.DEVNULL, stderr=sub.DEVNULL)  # , timeout=15)
    for i in range(90):
        if proc_status.poll() is not None:
            # print(f"Animatlab finished successfully.", flush=True)
            return
        else:
            time.sleep(1)
    # print(f"Animatlab timeout!!!", flush=True)
    # print(f"Animatlab timeout; returncode is {proc_status.returncode}", flush=True)
    return


def process_sim_data(path_asim_file: Path) -> dict:
    """
    process_sim_data(simpath) for all DataCharts in .asim file, creates Dataframe

    :param path_asim_file: file path to the standalone animatlab .asim file
    :return sim_data: a dictionary with keys being the name of DataCharts in from Animatlab.
                The "value" of each key is a DataFrame with the info from the Animatlab datachart.
                      sim_data { <key>: <value> }
    """
    sim_data = {}
    with open(path_asim_file, 'rb') as asim_file:
        asim_tree = ET.parse(asim_file)
        asim_root = asim_tree.getroot()
        for element in asim_root.findall('.//OutputFilename'):
            file_path = os.getcwd() + '\\' + element.text
            file_name = element.text.split('.txt')[0]
            with open(file_path, 'r') as path_text_file:
                sim_data[file_name] = pd.read_csv(path_text_file, sep="\t")

    return sim_data

def run_simulation(asim_path: Path, path_simulator: Path) -> float:

    print(f"Running Animatlab for {asim_path}...", flush=True)
    execute_simulation(asim_path, path_simulator)
    print(f"Processing data...", flush=True)
    sim_data = process_sim_data(asim_path)



def main():
    # path_simulator = Path.home() / "AnimatLab" / "bin" / "AnimatSimulator.exe"
    path_simulator = Path("C:/Program Files (x86)/NeuroRobotic Technologies/AnimatLab/bin/AnimatSimulator.exe")
    path_asim_file = Path("K&A synergy_Standalone.asim")

   
    run_simulation(path_asim_file, path_simulator)


if __name__ == '__main__':
    t = time.time()
    main()
    print('Time to complete: ', time.time() - t)

 # This file is all the basic functions to run an .asim file and evaluate its cost
 # Does not include any plotting
 # Also has input_new_parameters function which makes a new .asim file with a specified path


 # Basically the bare-bones of rat_hindlimb_python