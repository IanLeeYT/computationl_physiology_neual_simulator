import numpy as np
import matplotlib.pyplot as plt
import Neuron
import Synapse
import os


def save(in_vars):
    timestep, final_timestep, total_neuron_number, all_neurons, neuron_dict, group_names, all_connections, \
    connection_dict, connection_names, external_inputs, sniff_frequency, nclass_dict, save_info = in_vars
    load_dir, keep_load, save_dir = save_info

    if save_dir == "":
        print("no save made")
        return

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, "state.txt")
    file = open(file_name, "w")
    file.write("Simulator_setup:\n")
    file.write("timestep "+str(timestep)+"\n")
    file.write("final_timestep "+str(final_timestep)+"\n")
    file.write("load_directory "+str(save_dir)+"\n")
    file.write("keep_loaded_data 1\n")
    file.write("\n")
    file.write("Model_structure:\n")

    for group_name in group_names:
        lst = neuron_dict[group_name]
        cls_name = list(Neuron.neuron_names.keys())[list(Neuron.neuron_names.values()).index
                                                    (lst[0].__class__)]
        # print(group_name, cls_name)
        file.write("group "+cls_name+" "+group_name+" "+str(len(lst))+"\n")
    file.write("\n")

    for connection_name in connection_names:
        for syn in connection_dict[connection_name]:
            cls_name = (list(Synapse.synapse_names.keys())[list(Synapse.synapse_names.values()).index(syn.__class__)])
            pre = str(syn.pre.id)
            post = str(syn.post.id)
            file.write("connect "+connection_name+" "+cls_name+" {"+pre+"} {"+post+"} 1.0\n")
    file.write("\n")

    file.write("Parameters:\n")
    for neuron in all_neurons:
        keys = list(neuron.param_dict.keys())
        instr = "edit_neurons {"+str(neuron.id)+"} "
        for key in keys:
            instr = instr + key + " " + str(neuron.param_dict[key])
            if key != keys[-1]:
                instr = instr + ", "
        file.write(instr+"\n")
    file.write("\n")

    for connection_name in connection_names:
        syn = connection_dict[connection_name][0]
        keys = list(syn.param_dict.keys())
        instr = "edit_connection "+connection_name+" "
        for key in keys:
            instr = instr + key + " " + str(syn.param_dict[key])
            if key != keys[-1]:
                instr = instr + ", "
        file.write(instr+"\n")
    file.write("\n")

    voltage = np.zeros((len(all_neurons), final_timestep))
    output = np.zeros((len(all_neurons), final_timestep))
    i = 0
    for neuron in all_neurons:
        voltage[i, :] = np.copy(neuron.voltage_history)
        output[i, :] = np.copy(neuron.output_history)
        i += 1

    weights = np.zeros((len(all_connections), final_timestep))
    i = 0
    for syn in all_connections:
        weights[i, :] = np.copy(syn.weight_history)
        i += 1

    save_as_txt = True

    npz_name = os.path.join(save_dir, "data.npz")
    np.savez_compressed(npz_name, voltage=voltage, output=output, weights=weights)


    #file.write()

    file.close()
    print("save made:", file_name)


def load_data(in_vars):
    timestep, final_timestep, total_neuron_number, all_neurons, neuron_dict, group_names, all_connections, \
    connection_dict, connection_names, external_inputs, sniff_frequency, nclass_dict, save_info = in_vars
    load_dir, keep_load, save_dir = save_info

    if load_dir != "":
        npz_name = os.path.join(load_dir, "data.npz")
        data = np.load(npz_name)
        voltage = data["voltage"]
        output = data["output"]
        weights = data["weights"]

        print(keep_load, )
        if keep_load:
            final_timestep += voltage.shape[-1]
            starting = voltage.shape[-1]
            i = 0
            for neuron in all_neurons:
                neuron.voltage_history = np.zeros(final_timestep)
                neuron.voltage_history[:starting] = np.copy(voltage[i])
                neuron.output_history = np.zeros(final_timestep)
                neuron.output_history[:starting] = np.copy(output[i])
                i += 1
            i = 0
            for syn in all_connections:
                syn.weight_history = np.zeros(final_timestep)
                syn.weight_history[:starting] = np.copy(weights[i])
                i += 1
            print(starting, final_timestep)
            return starting, final_timestep
        else:
            i = 0
            for neuron in all_neurons:
                neuron.voltage_history[0] = voltage[i][-1]
                neuron.output_history[0] = output[i][-1]
                i += 1
            i = 0
            for syn in all_connections:
                syn.weight_history[0] = weights[i][-1]
                i += 1
            return 1, final_timestep
    else:
        return 1, final_timestep



