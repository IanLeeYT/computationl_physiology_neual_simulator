import Neuron
import Synapse
import numpy as np
import random
import sys
import os

if sys.version_info >= (3, 0):
    input_fun = input
else:
    input_fun = raw_input  # ignore Unresolved reference issue

line_no = 0
start_model = "Model_structure"
start_params = "Parameters"
start_output = "Output"

predef_output_lines = []

timestep = 1.0
final_timestep = 10
sniff_frequency = 10

load_dir = ""
keep_load = False
save_dir = ""

total_neuron_number = 0
all_neurons = []
neuron_dict = {}
nclass_dict = {}
group_names = []
all_connections = []
connection_dict = {}
connection_names = []
connection_name_class = {}

external_inputs = {}


def interpret_file(file_name):

    file = open(file_name, 'r')
    interpret_simulator_setup(file)
    interpret_model_structure(file)
    interpret_parameters(file)
    interpret_output_start(file)

    save_info = (load_dir, keep_load, save_dir)
    return timestep, final_timestep, total_neuron_number, all_neurons, neuron_dict, group_names, all_connections, \
        connection_dict, connection_names, external_inputs, sniff_frequency, nclass_dict, save_info


def interpret_simulator_setup(file):
    global line_no, timestep, final_timestep, sniff_frequency, load_dir, keep_load, save_dir

    line = file.readline()
    line_no += 1
    line = file.readline()
    line_no += 1
    while not (start_model in line):
        words = line.split()
        words = [x for x in words if x != '']

        if len(words) == 0 or words[0][0] == "#":
            pass
        elif words[0] == "timestep":
            inputs_ex(2, "timestep", words)
            timestep = float(words[1])
        elif words[0] == "final_timestep":
            inputs_ex(2, "final_timestep", words)
            final_timestep = int(words[1])
        elif words[0] == "final_time":
            inputs_ex(2, "final_time", words)
            final_timestep = int(float(words[1]) / timestep)
        elif words[0] == "sniff_frequency":
            inputs_ex(2, "sniff_frequency", words)
            sniff_frequency = float(words[1])
        elif words[0] == "load_directory":
            inputs_ex(2, "load_directory", words)
            load_dir = str(words[1])
        elif words[0] == "keep_loaded_data":
            inputs_ex(2, "keep_loaded_data", words)
            keep_load = int(words[1]) != 0
        elif words[0] == "save_directory":
            inputs_ex(2, "save_directory", words)
            save_dir = str(words[1])
        else:
            raise Exception("Invalid expression: {}".format(line))

        line = file.readline()
        line_no += 1


def interpret_model_structure(file):
    global line_no, total_neuron_number, all_neurons, neuron_dict, group_names,\
        all_connections, connection_dict, connection_names, final_timestep, nclass_dict, connection_name_class

    line = file.readline()
    line_no += 1
    while not (start_params in line):
        words = line.split()
        words = [x for x in words if x != '']

        if len(words) == 0 or words[0][0] == "#":
            pass
        elif words[0] == "group":
            neuron_type = words[1]
            group_name = words[2]
            group_size = int(words[3])
            lst = []
            for i in range(group_size):
                novel = Neuron.neuron_names[neuron_type](total_neuron_number + i, final_timestep)
                lst.append(novel)
                all_neurons.append(novel)
                if neuron_type in nclass_dict:
                    nclass_dict[neuron_type].append(novel)
                else:
                    nclass_dict[neuron_type] = [novel]
            total_neuron_number += group_size
            neuron_dict[group_name] = lst
            group_names.append(group_name)

        elif words[0] == "connect":
            connection_name = words[1]
            if "{" in words[2]:
                syn_type = "basicS"
            else:
                syn_type = words[2]
                words[2:-1] = words[3:]
                words = words[:-1]
            brace_string = "".join(words[2:])
            brace_string1 = brace_string.split("}")[0]
            brace_string1 = brace_string1.split("{")[1]
            brace_string2 = brace_string.split("}")[1]
            brace_string2 = brace_string2.split("{")[1]


            brace1 = brace_string1.split(",")
            brace2 = brace_string2.split(",")

            lst1 = interpret_brace(brace1)
            lst2 = interpret_brace(brace2)

            probstr = brace_string.split("}")[2]
            if probstr == "":
                prob = 1
            else:
                prob = float(probstr)

            connections = []
            for f in lst1:
                for t in lst2:
                    if f != t:
                        if random.random() <= prob:
                            connections.append(Synapse.synapse_names[syn_type]
                                               (all_neurons[f], all_neurons[t], final_timestep))
            connection_name_class[connection_name] = Synapse.synapse_names[syn_type]

            all_connections.extend(connections)
            if connection_name in list(connection_names):
                connection_dict[connection_name].extend(connections)
            else:
                connection_dict[connection_name] = connections
                connection_names.append(connection_name)

        elif words[0] == "connect_with_reps":
            connection_name = words[1]
            if "{" in words[2]:
                syn_type = "basicS"
            else:
                syn_type = words[2]
                words[2:-1] = words[3:]
                words = words[:-1]
            brace_string = "".join(words[2:])
            brace_string1 = brace_string.split("}")[0]
            brace_string1 = brace_string1.split("{")[1]
            brace_string2 = brace_string.split("}")[1]
            brace_string2 = brace_string2.split("{")[1]

            brace1 = brace_string1.split(",")
            brace2 = brace_string2.split(",")

            lst1 = interpret_brace(brace1)
            lst2 = interpret_brace(brace2)

            probstr = brace_string.split("}")[2]
            if probstr == "":
                prob = 1
            else:
                prob = float(probstr)

            connections = []
            for f in lst1:
                for t in lst2:
                    if random.random() <= prob:
                        connections.append(Synapse.synapse_names[syn_type]
                                           (all_neurons[f], all_neurons[t], final_timestep))
            connection_name_class[connection_name] = Synapse.synapse_names[syn_type]

            all_connections.extend(connections)
            if connection_name in list(connection_names):
                connection_dict[connection_name].extend(connections)
            else:
                connection_dict[connection_name] = connections
                connection_names.append(connection_name)

        elif words[0] == "connect_focus":
            connection_name = words[1]
            if "{" in words[2]:
                syn_type = "basicS"
            else:
                syn_type = words[2]
                words[2:-1] = words[3:]
                words = words[:-1]
            brace_string = "".join(words[2:])
            brace_string1 = brace_string.split("}")[0]
            brace_string1 = brace_string1.split("{")[1]
            brace_string2 = brace_string.split("}")[1]
            brace_string2 = brace_string2.split("{")[1]


            brace1 = brace_string1.split(",")
            brace2 = brace_string2.split(",")

            lst1 = interpret_brace(brace1)
            lst2 = interpret_brace(brace2)

            probstr = brace_string.split("}")[2]
            if probstr == "":
                prob = 1
            else:
                prob = float(probstr)

            connections = []
            for f in lst1:
                t = lst2[random.randint(0, len(lst2) - 1)]
                if f != t:
                    if random.random() <= prob:
                        connections.append(Synapse.synapse_names[syn_type]
                                           (all_neurons[f], all_neurons[t], final_timestep))
            connection_name_class[connection_name] = Synapse.synapse_names[syn_type]

            all_connections.extend(connections)
            if connection_name in list(connection_names):
                connection_dict[connection_name].extend(connections)
            else:
                connection_dict[connection_name] = connections
                connection_names.append(connection_name)

        elif words[0] == "connect_one_to_one":
            connection_name = words[1]
            if "{" in words[2]:
                syn_type = "basicS"
            else:
                syn_type = words[2]
                words[2:-1] = words[3:]
                words = words[:-1]
            brace_string = "".join(words[2:])
            brace_string1 = brace_string.split("}")[0]
            brace_string1 = brace_string1.split("{")[1]
            brace_string2 = brace_string.split("}")[1]
            brace_string2 = brace_string2.split("{")[1]


            brace1 = brace_string1.split(",")
            brace2 = brace_string2.split(",")

            lst1 = interpret_brace(brace1)
            lst2 = interpret_brace(brace2)

            probstr = brace_string.split("}")[2]
            if probstr == "":
                prob = 1
            else:
                prob = float(probstr)

            if len(lst1) != len(lst2):
                raise Exception("Lists must have the same length for connect_one_to_one")

            connections = []
            for ct in range(len(lst1)):
                f = lst1[ct]
                t = lst2[ct]
                connections.append(Synapse.synapse_names[syn_type](all_neurons[f], all_neurons[t], final_timestep))
            connection_name_class[connection_name] = Synapse.synapse_names[syn_type]

            all_connections.extend(connections)
            if connection_name in list(connection_names):
                connection_dict[connection_name].extend(connections)
            else:
                connection_dict[connection_name] = connections
                connection_names.append(connection_name)

        else:
            raise Exception("Invalid expression: {}".format(line))

        line = file.readline()
        line_no += 1


def interpret_parameters(file):
    global line_no, timestep, final_timestep, all_neurons, connection_dict, external_inputs

    line = file.readline()
    line_no += 1
    while (start_output not in line) and ("\n" in line or line != ""):
        words = line.split()
        words = [x for x in words if x != '']

        if len(words) == 0 or words[0][0] == "#":
            pass
        elif words[0] == "edit_neurons":
            brace_string = "".join(words[1:])
            brace_string = brace_string.split("}")[0]
            brace_string = brace_string.split("{")[1]
            brace = brace_string.split(",")
            lst = interpret_brace(brace)
            start = [("}" in s) for s in words].index(True)
            remaining = words[start + 1:]
            remaining = [s.replace(",", "") for s in remaining]
            remaining = [x for x in remaining if x != ""]

            while len(remaining) > 0:
                parameter_name, val, remaining = interpret_subparams(remaining, len(lst))

                for i in range(len(lst)):
                    if parameter_name not in all_neurons[lst[i]].param_dict:
                        raise Exception("Invalid connection parameter name: {}".format(parameter_name))
                    else:
                        all_neurons[lst[i]].param_dict[parameter_name] = val[i]

        elif words[0] == "edit_connection":
            connection_name = words[1]
            remaining = words[2:]
            remaining = [s.replace(",","") for s in remaining]
            remaining = [x for x in remaining if x != ""]

            while len(remaining) > 0:
                parameter_name, val, remaining = interpret_subparams(remaining, len(connection_dict[connection_name]))
                if parameter_name not in (connection_dict[connection_name][0]).param_dict:
                    raise Exception("Invalid connection parameter name: {}".format(parameter_name))
                for i in range(len(connection_dict[connection_name])):
                    (connection_dict[connection_name][i]).param_dict[parameter_name] = val[i]
        elif words[0] == "external_stimulus" or words[0] == "estim":
            brace_string = "".join(words[1:])
            brace_string1 = brace_string.split("}")[0]
            brace_string1 = brace_string1.split("{")[1]
            brace_string2 = brace_string.split("}")[1]
            brace_string2 = brace_string2.split("{")[1]

            brace1 = brace_string1.split(",")
            brace2 = brace_string2.split(",")

            n_list = interpret_brace(brace1)
            times = interpret_integer_brace(brace2)

            val = float(brace_string.split("}")[2])
            for i in n_list:
                for j in times:
                    if (i, j) in external_inputs:
                        external_inputs[(i, j)] += val
                    else:
                        external_inputs[(i, j)] = val

        elif words[0] == "ostim":
            brace_string = "".join(words[1:])
            brace_string1 = brace_string.split("}")[0]
            brace_string1 = brace_string1.split("{")[1]
            brace_string2 = brace_string.split("}")[1]
            brace_string2 = brace_string2.split("{")[1]

            brace1 = brace_string1.split(",")
            brace2 = brace_string2.split(",")

            n_list = interpret_brace(brace1)
            times = interpret_integer_brace(brace2)

            start = [("}" in s) for s in words].index(True)
            start = [("}" in s) for s in words].index(True, start+1)

            remaining = words[start + 1:]

            max_val = float(remaining[0])
            for_time = int(remaining[1])
            mean = int(remaining[2])
            std = float(remaining[3])

            for i in n_list:
                for j in times:
                    for tl in range(for_time):
                        if (i, j + tl) in external_inputs:
                            external_inputs[(i, j + tl)] += max_val *\
                                                            np.exp(-0.5 * np.power((i - mean) / std, 2))
                        else:
                            external_inputs[(i, j + tl)] = max_val * \
                                                            np.exp(-0.5 * np.power((i - mean) / std, 2))
        elif words[0] == "sniff":
            brace_string = "".join(words[1:])
            brace_string1 = brace_string.split("}")[0]
            brace_string1 = brace_string1.split("{")[1]
            brace_string2 = brace_string.split("}")[1]
            brace_string2 = brace_string2.split("{")[1]

            brace1 = brace_string1.split(",")
            brace2 = brace_string2.split(",")

            n_list = interpret_brace(brace1)
            times = interpret_integer_brace(brace2)

            start = [("}" in s) for s in words].index(True)
            start = [("}" in s) for s in words].index(True, start+1)

            remaining = words[start + 1:]

            max_val = float(remaining[0])
            freq = int(remaining[1])
            mean = int(remaining[2])
            std = float(remaining[3])

            for i in n_list:
                for j in times:
                    temp_val = max_val * np.exp(-0.5 * np.power((i - mean) / std, 2)) *\
                           ((1 - np.cos(2. * np.pi * (j - times[0]) * freq / 1000.)) / 2.)
                    if (i, j) in external_inputs:
                        external_inputs[(i, j)] += temp_val
                    else:
                        external_inputs[(i, j)] = temp_val

        else:
            raise Exception("Invalid expression: {}".format(line))

        line = file.readline()
        line_no += 1


def interpret_output_start(file):
    global line_no, timestep, final_timestep, all_neurons, connection_dict, external_inputs

    line = file.readline()
    line_no += 1
    while "\n" in line or line != "":
        predef_output_lines.append(line)
        line = file.readline()
        line_no += 1


def interpret_subparams(remaining, length):
    parameter_name = remaining[0]
    val = []
    try:
        val = [float(remaining[1])] * length
        used = 2
    except ValueError:
        if remaining[1] == "uniform":
            a = float(remaining[2])
            b = float(remaining[3])
            val = list(np.random.uniform(a, b, length))
            used = 4
        elif remaining[1] == "normal" or remaining[1] == "gaussian":
            mean = float(remaining[2])
            std = float(remaining[3])
            val = list(np.random.normal(mean, std, length))
            used = 4
        else:
            raise Exception("Invalid probability distribution for parameter: {}".format(parameter_name))
    return parameter_name, val, remaining[used:]


def interpret_brace(words):
    global total_neuron_number, neuron_dict
    lst = []

    for word in words:
        if ":" in word:
            itr = word.split(":")
            if len(itr) == 2:
                start = 0 if itr[0] == "" else int(itr[0])
                fin = total_neuron_number if itr[1] == "" else int(itr[1])
                lst.extend(list(np.arange(start, fin)))
            elif len(itr) == 3:
                start = 0 if itr[0] == "" else int(itr[0])
                fin = total_neuron_number if itr[1] == "" else int(itr[1])
                step = 1 if itr[2] == "" else int(itr[2])
                lst.extend(list(np.arange(start, fin, step)))
            else:
                raise Exception("Incorrect syntax for start:stop:step notation")

        else:
            try:
                val = [int(word)]
            except ValueError:
                val = [neuron.id for neuron in neuron_dict[word]]
            lst.extend(val)

    lst = list(dict.fromkeys(lst))
    lst.sort()
    return lst


def interpret_integer_brace(words):
    global final_timestep
    lst = []

    for word in words:
        if ":" in word:
            itr = word.split(":")
            if len(itr) == 2:
                start = 0 if itr[0] == "" else int(itr[0])
                fin = final_timestep if itr[1] == "" else int(itr[1])
                lst.extend(list(np.arange(start, fin)))
            elif len(itr) == 3:
                start = 0 if itr[0] == "" else int(itr[0])
                fin = final_timestep if itr[1] == "" else int(itr[1])
                step = 1 if itr[2] == "" else int(itr[2])
                lst.extend(list(np.arange(start, fin, step)))
            else:
                raise Exception("Incorrect syntax for start:stop:step notation")

        else:
            val = [int(word)]
            lst.extend(val)

    lst = list(dict.fromkeys(lst))
    lst.sort()
    return lst


def inputs_ex(num, key, words):
    if len(words) != num:
        raise Exception("Wrong number of inputs for {}".format(key))


def get_filename():
    ye, no = ["y", "yes", "1", "Y"], ["n", "no", "0", "N"]
    meta_data = "metadata.txt"
    if os.path.exists(meta_data):
        md = open(meta_data, "r")
        d = md.readlines()
        d = [x.rstrip() for x in d]
        md.close()
        if d[0] == "force":
            fn = d[1]
            print(fn)
            if os.path.exists(fn):
                print("Using forced model from "+str(fn))
                return fn
            else:
                raise Exception("Forced file name does not exist. Please edit or delete \"metadata.txt\".")
        else:
            fn = d[0]
            if os.path.exists(fn):
                use = input_fun(("Should the last used model from "+str(fn)+" be loaded? y/n:\n"))
                while use not in ye and use not in no:
                    use = input_fun(("INVALID INPUT: Should the model from " + str(fn) + " be loaded? y/n:\n"))
                if use in ye:
                    return fn
                elif use in no:
                    fn = input_fun("Enter new file name/path:\n")
                    while not os.path.exists(fn):
                        print("The file or path entered does not exist")
                        fn = input_fun("INVALID INPUT: Enter new file name/path:\n")
                    md = open(meta_data, "w")
                    md.write(fn)
                    md.close()
                    return fn
                else:
                    raise Exception("INPUT BROKEN: CONTACT CODE WRITER - krm74@cornell.edu")
            else:
                raise Exception("Old file name does not exist. Please edit or delete \"metadata.txt\".")
    else:
        fn = input_fun("Enter new file name/path:\n")
        while not os.path.exists(fn):
            print("The file or path entered does not exist")
            fn = input_fun("INVALID INPUT: Enter new file name/path:\n")
        md = open(meta_data, "w")
        md.write(fn)
        md.close()
        return fn


def get_save_selection(line=-1):
    if line == -1:
        print("Format: <plot/save> neuron <data_name> <func> {<Neurons>} {<times>}")
        print("or: <plot/save> synapse <data_name> <func> {<Neurons from>} {<Neurons to>} {<times>}")
        print("or: <plot/save> connection <data_name> <func> <connection name> {<times>}")
        sel = input_fun("enter save or plot selection, or \"end\"\n")
    else:
        sel = line
    sp = sel.split()
    sp = [x for x in sp if x != ""]
    if len(sp) == 0 or sp[0][0] == "#":
        return None
    elif sp[0] == "end":
        return "end"
    elif sp[1] == "neuron":
        brace_string = "".join(sp[4:])
        brace_string1 = brace_string.split("}")[0]
        brace_string1 = brace_string1.split("{")[1]
        brace_string2 = brace_string.split("}")[1]
        brace_string2 = brace_string2.split("{")[1]
        brace1 = brace_string1.split(",")
        brace2 = brace_string2.split(",")

        n_list = interpret_brace(brace1)
        neuron_list = [all_neurons[x] for x in n_list]
        # print([n.id for n in neuron_list])
        times = interpret_integer_brace(brace2)

        grp_name = brace_string1
        return sp[0], sp[1], sp[2], sp[3], neuron_list, times, grp_name

    elif sp[1] == "synapse":
        brace_string = "".join(sp[4:])
        brace_string1 = brace_string.split("}")[0]
        brace_string1 = brace_string1.split("{")[1]
        brace_string2 = brace_string.split("}")[1]
        brace_string2 = brace_string2.split("{")[1]
        brace_string3 = brace_string.split("}")[2]
        brace_string3 = brace_string3.split("{")[1]
        brace1 = brace_string1.split(",")
        brace2 = brace_string2.split(",")
        brace3 = brace_string3.split(",")
        from_list = interpret_brace(brace1)
        to_list = interpret_brace(brace2)
        times = interpret_integer_brace(brace3)
        c_list = []
        for syn in all_connections:
            if syn.pre.id in from_list and syn.post.id in to_list:
                c_list.append(syn)
        grp_name = "from "+brace_string1+" to "+brace_string2
        return sp[0], sp[1], sp[2], sp[3], c_list, times, grp_name

    elif sp[1] == "connection":
        c_name = sp[4]
        brace_string = "".join(sp[5:])
        brace_string1 = brace_string.split("}")[0]
        brace_string1 = brace_string1.split("{")[1]
        brace1 = brace_string1.split(",")
        times = interpret_integer_brace(brace1)
        c_list = connection_dict[c_name]
        return sp[0], sp[1], sp[2], sp[3], c_list, times, c_name

    else:
        print("Error: invalid input")
        return None


if __name__ == "__main__":

    file_name = "test2.txt"
    interpret_file(file_name)
    print(group_names)
    for name in group_names:
        print([neuron.id for neuron in neuron_dict[name]])

    print(connection_names)
    for name in connection_names:
        print(connection_dict[name])

    print(line_no)
