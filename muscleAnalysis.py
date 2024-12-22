# --------------------------------------------------------------------------- #
# OpenSim: addPathSpring.py                                                   #
# --------------------------------------------------------------------------- #
# OpenSim is a toolkit for musculoskeletal modeling and simulation,           #
# developed as an open source project by a worldwide community. Development   #
# and support is coordinated from Stanford University, with funding from the  #
# U.S. NIH and DARPA. See http://opensim.stanford.edu and the README file     #
# for more information including specific grant numbers.                      #
#                                                                             #
# Copyright (c) 2005-2017 Stanford University and the Authors                 #
# Author(s): Ayman Habib                                                      #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License"); you may     #
# not use this file except in compliance with the License. You may obtain a   #
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0           #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
# --------------------------------------------------------------------------- #
# Get a handle to the current model and create a new copy
# # Get a handle to the current model and create a new copy 

import opensim as osim
import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors
import re  # Importing for regex
import csv


def get_folder_number(root, base_path):
    """
    Extract the folder number from the current root path.

    Parameters:
        root (str): Current directory path from os.walk.
        base_path (str): Base path where os.walk starts.

    Returns:
        int: The folder number (integer value of the folder name).
    """
    relative_path = os.path.relpath(root, base_path)
    folder_name = relative_path.split(os.sep)[0]
    try:
        return int(folder_name)  # Convert to integer
    except ValueError:
        return None

def extract_folder_number(folder_name):
    """
    Extract the numerical portion of the folder name.
    """
    try:
        return int(folder_name)
    except ValueError:
        return None

def plot_moment_product(static_opt_path, muscle_analysis_path, muscle, output_path,Mtype,inputPart):
    """
    Plot the product of moment arm values and force values over the gait cycle.

    Parameters:
        static_opt_path (str): Path to the StaticOptimization folders.
        muscle_analysis_path (str): Path to the MuscleAnalysis folders.
        muscle (str): Name of the muscle to analyze.
        output_path (str): Path to save the plot.
    """
    # Collect folders from both paths
    static_folders = {
        extract_folder_number(folder): os.path.join(static_opt_path, folder)
        for folder in os.listdir(static_opt_path)
        if os.path.isdir(os.path.join(static_opt_path, folder))
    }
    muscle_analysis_folders = {
        extract_folder_number(folder): os.path.join(muscle_analysis_path, folder)
        for folder in os.listdir(muscle_analysis_path)
        if os.path.isdir(os.path.join(muscle_analysis_path, folder))
    }

    # Get the common folder numbers
    common_folders = sorted(set(static_folders.keys()) & set(muscle_analysis_folders.keys()))

    if not common_folders:
        print("No matching folders found between the two paths.")
        return

    # Initialize plot
    plt.figure(figsize=(10, 6))
    colors = create_color_gradient(len(common_folders))

    for folder_number, color in zip(common_folders, colors):
        static_folder = static_folders[folder_number]
        muscle_folder = muscle_analysis_folders[folder_number]

        # Load force file from StaticOptimization path
        force_file = next(
            (os.path.join(static_folder, file) for file in os.listdir(static_folder) if file.endswith("force.sto")),
            None
        )
        if not force_file:
            print(f"Force file not found in folder: {static_folder}")
            continue

        # Load moment arm file from MuscleAnalysis path
        moment_arm_file = os.path.join(muscle_folder, "AnalisisUI_MuscleAnalysis_MomentArm_ankle_angle_r.sto")
        if not os.path.isfile(moment_arm_file):
            print(f"Moment arm file not found in folder: {muscle_folder}")
            continue

        # Read the files into DataFrames
        try:
            with open(force_file, 'r') as f:
                # Skip header lines until 'endheader'
                while True:
                    line = f.readline()
                    if "endheader" in line:
                        break
                force_df = pd.read_csv(f, delim_whitespace=True)

            with open(moment_arm_file, 'r') as f:
                # Skip header lines until 'endheader'
                while True:
                    line = f.readline()
                    if "endheader" in line:
                        break
                moment_arm_df = pd.read_csv(f, delim_whitespace=True)
        except Exception as e:
            print(f"Error reading files in folder {folder_number}: {e}")
            continue

        # Check if the muscle exists in both files
        if muscle not in force_df.columns or muscle not in moment_arm_df.columns:
            print(f"Muscle '{muscle}' not found in files for folder {folder_number}.")
            continue

        # Compute the product of force and moment arm
        product = -1 * force_df[muscle] * moment_arm_df[muscle]

        # Normalize x-axis to percentage (0-100)
        x_values = (np.arange(len(product)) / (len(product) - 1)) * 100

        # Add the line to the plot
        if inputPart == "Part1":
            plt.plot(x_values, product, label=f"{100 - folder_number}%", color=color)
        else:
            plt.plot(x_values, product, label=f"{folder_number}%", color=color)

    # Customize plot
    plt.title(f"{muscle} Moment Over Gait Cycle")
    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Moment")
    legend_title = "Soleus at strength:" if Mtype == "M1" else "Gastrocnemius at strength:"
    plt.legend(title=legend_title, loc='upper right')
    plt.grid(True)
    
    # Save the plot
    filePath = os.path.join(output_path, f"{folder_number}_{muscle}_moments.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(filePath, format='png', dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")

def plotForMuscle(folder_path, muscle, Mtype, part):
    # Dictionary to store activations by folder number
    muscle_activations = {}

    # Traverse folders and collect one activation file per folder
    for root, _, files in os.walk(folder_path):
        folder_number = get_folder_number(root, folder_path)
        if folder_number is None:
            continue  # Skip folders without valid numbers

        for file in files:
            if file.endswith("activation.sto"):
                sto_path = os.path.join(root, file)

                # Read the .sto file
                with open(sto_path, 'r') as f:
                    # Skip header lines until 'endheader'
                    while True:
                        line = f.readline()
                        if "endheader" in line:
                            break
                    # Load the data into a DataFrame
                    df = pd.read_csv(f, delim_whitespace=True)

                # Check if the muscle exists
                if muscle in df.columns:
                    muscle_activations[folder_number] = df[muscle]
                else:
                    print(f"Muscle '{muscle}' not found in file: {sto_path}")
                break  # Only process one activation file per folder

    if not muscle_activations:
        print(f"No valid activation data found for muscle '{muscle}'.")
        return

    # Sort activations by folder number
    sorted_activations = sorted(muscle_activations.items())

    # Generate colors for the lines
    n_lines = len(sorted_activations)
    colors = create_color_gradient(n_lines)

    # Plotting
    plt.figure(figsize=(10, 6))
    for (folder_number, activations), color in zip(sorted_activations, colors):
        # Normalize x-axis to percentage (0-100)
        x_values = (np.arange(len(activations)) / (len(activations) - 1)) * 100
        if part == "Part1":
            label = f"{100 - folder_number}%"  # E.g., 100% strength
        else:
            label = f"{folder_number}%"  # E.g., folder strength

        plt.plot(x_values, activations, label=label, color=color)

    # Customize plot
    plt.title(f"{muscle} Activation Over Gait Cycle")
    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Activation")
    legend_title = "Soleus at strength:" if Mtype == "M1" else "Gastrocnemius at strength:"
    plt.legend(title=legend_title, loc='upper right')
    plt.grid(True)

    # Save plot
    output_path = f"Solutions/StaticOptimization/{part}/{Mtype}/Plots/{muscle}ActivationOverWeakness.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()  # Close plot to avoid overlap

def findTopActivationVariation(muscles, base_folder_path,part):
    # Step 1: Remove duplicate muscles
    muscles = list(set(muscles))
    
    # Initialize a dictionary to store variance data for each muscle
    muscle_variances = {muscle: [] for muscle in muscles}
    
    # Step 2: Iterate through folders 0, 20, 40, 60, 80, 100
    if part == "Part1":
        folder_indices = [0, 20, 40, 60, 80, 100]
    else:
        folder_indices = [120, 140, 160]
    for i in folder_indices:
        folder_path = os.path.join(base_folder_path, str(i))
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue
        
        # Find the activation file in the current folder
        sto_files = [f for f in os.listdir(folder_path) if f.endswith('activation.sto')]
        if not sto_files:
            print(f"No activation file found in {folder_path}. Skipping.")
            continue
        
        sto_path = os.path.join(folder_path, sto_files[0])  # Assume one file per folder
        
        # Load the .sto file as a DataFrame
        with open(sto_path, 'r') as file:
            # Skip header lines until the column names
            while True:
                line = file.readline()
                if 'endheader' in line:
                    break
            # Read the data into a DataFrame
            df = pd.read_csv(file, delim_whitespace=True)
        
        # Calculate variance for each muscle in the list
        for muscle in muscles:
            if muscle in df.columns:
                variance = df[muscle].var()
                muscle_variances[muscle].append(variance)
    
    # Step 3: Calculate the total variance across files for each muscle
    total_variance = {}
    for muscle, variances in muscle_variances.items():
        if len(variances) == len(folder_indices):  # Ensure all files contributed
            total_variance[muscle] = sum(variances)
    
    # Step 4: Identify the top 5 muscles with the highest total variance
    top_muscles = sorted(total_variance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Step 5: Return the names of the top 5 muscles
    return [muscle for muscle, _ in top_muscles]

def create_color_gradient(n_colors, start_color=(1.0, 0.0, 0.0), end_color=(0.0, 0.0, 1.0)):
    start_color_arr = np.array(start_color)
    end_color_arr = np.array(end_color)
    return np.linspace(start_color_arr, end_color_arr, n_colors)

def changeXml(file_path, new_value):
    tree = ET.parse(file_path)
    root = tree.getroot()
    model_file_tag = root.find(".//model_file")
    if model_file_tag is not None:
        model_file_tag.text = new_value
    else:
        raise ValueError("The 'model_file' tag was not found in the XML.")
      # Save the changes back to the same file
    tree.write(file_path, encoding="unicode", xml_declaration=True)

def muscleAnalysisTool(setup_file, output_directory, value, muscleType, part):
    try:
        # Create the custom folder inside the output directory
        custom_folder = os.path.join(output_directory, value)
        os.makedirs(custom_folder, exist_ok=True)

        changeXml(setup_file, f"Solutions/MuscleAnlysis/{part}/{muscleType}/Models/subject_scaled_RRA2_MW_{value}Perc.osim")

        # Load the setup file
        analysis_tool = osim.AnalyzeTool(setup_file)

        # Update the results directory to the custom folder
        analysis_tool.setResultsDir(custom_folder)

        # Run the analysis
        analysis_tool.run()
        print("Muscle analysis completed successfully.")
    except Exception as e:
        print(f"Error during muscle analysis: {e}")

def staticOptimizationTool(setup_file, output_directory, value, muscleType, part):
    try:
        # Create the custom folder inside the output directory
        custom_folder = os.path.join(output_directory, value)
        os.makedirs(custom_folder, exist_ok=True)

        changeXml(setup_file, f"Solutions/StaticOptimization/{part}/{muscleType}/Models/subject_scaled_RRA2_MW_{value}Perc.osim")

        # Load the setup file
        analysis_tool = osim.AnalyzeTool(setup_file)

        # Update the results directory to the custom folder
        analysis_tool.setResultsDir(custom_folder)
        # Run the analysis
        analysis_tool.run()
        print("Muscle analysis completed successfully.")
    except Exception as c:
        print(f"Error during muscle analysis: {c}")

def Part1(muscle_type):
    # Load the OpenSim model
    model = osim.Model("Model/subject_scaled_RRA.osim")
    # Clone the model to work on a new instance
    newModel = model.clone()

    if muscle_type == "M1":
        # M1: Soleus muscle
        muscle_name = "soleus_r"
        muscles = [newModel.getMuscles().get(muscle_name)]
        prefix = "M1"
    elif muscle_type == "M2":
        # M2: Medial and Lateral gastrocnemius muscles
        muscle_names = ["med_gas_r", "lat_gas_r"]
        muscles = [newModel.getMuscles().get(name) for name in muscle_names]
        prefix = "M2"
    else:
        raise ValueError("Invalid muscle type. Please specify 'M1' or 'M2'.")

    # Retrieve the current maximum isometric force for each muscle
    original_strengths = [muscle.getMaxIsometricForce() for muscle in muscles]

    # Steps to reduce strength
    reduction_steps = [0,0.2,0.4,0.6,0.8,1.0]

    for step in reduction_steps:
        # Reduce strength for each muscle by the specified percentage
        new_strengths = [original * (1 - step) for original in original_strengths]
        for muscle, new_strength in zip(muscles, new_strengths):
            muscle.setMaxIsometricForce(new_strength)

        # Print the new strength for verification
        for i, new_strength in enumerate(new_strengths):
            print(f"{muscles[i].getName()} strength reduced to: {new_strength}")

        # Save the modified model
        modifiedModel = osim.Model(newModel)
        
        fullMAPathName = f"Solutions/MuscleAnlysis/Part1/{muscle_type}/Models/subject_scaled_RRA2_MW_{int(step * 100)}Perc.osim"
        modifiedModel.printToXML(fullMAPathName)

        fullSOPathName = f"Solutions/StaticOptimization/Part1/{muscle_type}/Models/subject_scaled_RRA2_MW_{int(step * 100)}Perc.osim"
        modifiedModel.printToXML(fullSOPathName)
        

        # Perform static optimization or other processing
        muscleAnalysisTool("MuscleAnalysisSetup.xml",f"Solutions/MuscleAnlysis/Part1/{muscle_type}/MA",str(int(step * 100)),muscle_type,"Part1")
        staticOptimizationTool("SO_setup_walkingNormal.xml",f"Solutions/StaticOptimization/Part1/{muscle_type}/SO",str(int(step * 100)),muscle_type,"Part1")


def Part2(muscle_type):
    # Load the OpenSim model
    model = osim.Model("Model/subject_scaled_RRA.osim")
    # Clone the model to work on a new instance
    newModel = model.clone()

    if muscle_type == "M1":
        # Keep med_gas_r and lat_gas_r at 0.7 of their original strength
        mantainedMuscleNames = ["med_gas_r", "lat_gas_r"]
        variationMuscleNames = ["soleus_r"]
    elif muscle_type == "M2":
        # Keep soleus_r at 0.7 of its original strength
        mantainedMuscleNames = ["soleus_r"]
        variationMuscleNames = ["med_gas_r", "lat_gas_r"]
    else:
        raise ValueError("Invalid muscle type. Please specify 'M1' or 'M2'.")

    # Retrieve muscle objects
    mantainedMuscles = [newModel.getMuscles().get(name) for name in mantainedMuscleNames]
    variationMuscles = [newModel.getMuscles().get(name) for name in variationMuscleNames]

    # Retrieve original strengths
    mantainedOriginalStrengths = [muscle.getMaxIsometricForce() for muscle in mantainedMuscles]
    variationOriginalStrengths = [muscle.getMaxIsometricForce() for muscle in variationMuscles]

    # Reduce maintained muscles to 0.7 of their original strength
    newMantainedStrengths = [original * 0.7 for original in mantainedOriginalStrengths]
    for muscle, newStrength in zip(mantainedMuscles, newMantainedStrengths):
        muscle.setMaxIsometricForce(newStrength)
        print(f"{muscle.getName()} strength set to: {newStrength}")

    # Steps to increase the strength of variation muscles
    variation_steps = [1.2, 1.4, 1.6]

    for step in variation_steps:
        # Increase the strength of the variation muscles
        newVariationStrengths = [original * step for original in variationOriginalStrengths]
        for muscle, newStrength in zip(variationMuscles, newVariationStrengths):
            muscle.setMaxIsometricForce(newStrength)

        # Print the new strength for verification
        for i, newStrength in enumerate(newVariationStrengths):
            print(f"{variationMuscles[i].getName()} strength increased to: {newStrength}")

        # Save the modified model
        modifiedModel = osim.Model(newModel)
        
        fullMAPathName = f"Solutions/MuscleAnlysis/Part2/{muscle_type}/Models/subject_scaled_RRA2_MW_{int(step * 100)}Perc.osim"
        modifiedModel.printToXML(fullMAPathName)

        fullSOPathName = f"Solutions/StaticOptimization/Part2/{muscle_type}/Models/subject_scaled_RRA2_MW_{int(step * 100)}Perc.osim"
        modifiedModel.printToXML(fullSOPathName)
        

        # Perform static optimization or other processing
        muscleAnalysisTool("MuscleAnalysisSetup.xml",f"Solutions/MuscleAnlysis/Part2/{muscle_type}/MA",str(int(step * 100)),muscle_type,"Part2")
        staticOptimizationTool("SO_setup_walkingNormal.xml",f"Solutions/StaticOptimization/Part2/{muscle_type}/SO",str(int(step * 100)),muscle_type,"Part2")


inputPart = input("Do you want to do Part1 or Part2?\n")
inputMuscle = input("Do you want to use M1 or M2?\n")
printMuscles = input("Do you want to print a specific muscle?\n")
# if inputPart == "Part1":
#     Part1(inputMuscle)
# else:
#     Part2(inputMuscle)


Pmuscles = ['tib_ant_r', 'ext_dig_r', 'ext_hal_r', 'per_tert_r', 'med_gas_r', 'soleus_r', 
            'tib_post_r', 'per_brev_r', 'per_long_r', 'semiten_r', 'bifemlh_r', 'sar_r', 
            'rect_fem_r', 'vas_lat_r', 'vas_int_r', 'glut_max1_r', 'glut_max2_r', 'glut_max3_r', 
            'bifemsh_r', 'semimem_r', 'grac_r', 'rect_fem_r', 'iliacus_r', 'psoas_r', 'sar_r']

#plotMuscles = ['soleus_r', 'med_gas_r', 'tib_post_r', 'per_long_r', 'per_brev_r']

folder_path = f"Solutions/StaticOptimization/{inputPart}/{inputMuscle}/SO"


print(findTopActivationVariation(Pmuscles,folder_path,inputPart))


#PLOT ACTIVATIONS
if printMuscles != ""
    plotForMuscle(folder_path, printMuscles, inputMuscle ,inputPart)  # Call your function
else:
    for muscle_name in plotMuscles:
        print(f"Plotting activation for: {muscle_name}")
        plotForMuscle(folder_path, muscle_name, inputMuscle ,inputPart)  # Call your function



#PLOT MOMENTS
if printMuscles != ""
        plot_moment_product(f"Solutions/StaticOptimization/{inputPart}/{inputMuscle}/SO",
            f"Solutions/MuscleAnlysis/{inputPart}/{inputMuscle}/MA", 
            printMuscles,
            f"Solutions/Moments/{inputPart}/{inputMuscle}",inputMuscle,inputPart)

else:
    for muscle_name in plotMuscles:
        plot_moment_product(f"Solutions/StaticOptimization/{inputPart}/{inputMuscle}/SO",
            f"Solutions/MuscleAnlysis/{inputPart}/{inputMuscle}/MA", 
            muscle_name,
            f"Solutions/Moments/{inputPart}/{inputMuscle}",inputMuscle,inputPart)

