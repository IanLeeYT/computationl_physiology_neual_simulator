import Interpreter
import numpy as np
import matplotlib.pyplot as plt
import time
import saver

import os
import scipy.stats as stts
import sys

if sys.version_info >= (3, 0):
    python_verion_3 = True
    import xlsxwriter as xw
else:
    python_verion_3 = False
    import pandas as pd
    import csv

if python_verion_3:
    print("python version 3")
else:
    print("python version 2")

n_plots = 0


def raster():
    # plots a raster plot (spike timing plot) for all neurons. colors different groups different colors
    fires = [np.greater(np.array(all_neurons[i].output_history), 0.1) for i in range(total_neuron_number)]
    fire_times = [[j for j in range(len(fires[i])) if fires[i][j]] for i in range(total_neuron_number)]
    fig, axs = plt.subplots(1, 1)
    axs.eventplot(fire_times, linelengths=0.5, colors="black")
    plt.ylabel("Neuron number")
    plt.xlabel("timestep")
    plt.title("raster")


def sub_raster(neurons, times):
    fires = [np.greater(np.array(neurons[i].output_history), 0.1) for i in range(len(neurons))]
    fire_times = [[j for j in range(len(fires[i])) if fires[i][j]] for i in range(len(neurons))]
    fire_times = [[t for t in ts if t in times] for ts in fire_times]
    # if not_first_plot:
    #     plt.figure()
    fig, axs = plt.subplots(1, 1)
    plt.xlim((np.min(times), np.max(times)))
    axs.eventplot(fire_times, linelengths=0.5, colors="black")
    plt.ylabel("Neuron")
    plt.xlabel("timestep")
    plt.title("raster")


def plt_subspec(val_lst, times, use_name):
    v = np.zeros(final_timestep)

    for vs in val_lst:
        v += vs
    v = v[times]
    # f, t, Sxx = signal.spectrogram(x/100, fs = 1000)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    nfft = 100
    noverlap = 85
    if not_first_plot:
        plt.figure()
    plt.specgram(v/len(val_lst), Fs=1000./timestep, NFFT=nfft, noverlap=noverlap)
    plt.xlabel('Time[s]')
    plt.ylabel('Frequency[Hz]')
    plt.title('Spectrogram of '+use_name)
    plt.ylim((0, 150))
    plt.tight_layout(pad=1.5)


if python_verion_3:
    def save_data_file3(X, Y, X_unit, name, nos, keyword, func, use_name, ids=None):
        """
        neuron: list of neuron types desired
        indices: tuple/list of tuples that indicates the starting and ending neurons
        times: tuple of start and end times of recording

        writes data into an excel file
        """

        save_xlsx_file_name = name
        og, cc = save_xlsx_file_name, 1
        while os.path.isfile(save_xlsx_file_name + '.xlsx'):
            save_xlsx_file_name = og + str(cc)
            cc += 1
        save_xlsx_file_name = save_xlsx_file_name + '.xlsx'
        workbook = xw.Workbook(save_xlsx_file_name)
        worksheet = workbook.add_worksheet()

        row_number, col_number = 0, 0
        title = "The following data is "+func+" of "+keyword+" from "+nos+"s "+use_name
        comment1 = "Row headers are "+nos+" and their indices."
        comment2 = "Column headers are "+X_unit
        worksheet.write(row_number, col_number, title)
        row_number += 1
        worksheet.write(row_number, col_number, comment1)
        row_number += 1
        worksheet.write(row_number, col_number, comment2)

        row_number, col_number = 4, 1
        for xi in range(len(X)):
            worksheet.write(row_number, col_number, X[xi])
            col_number += 1

        row_number, col_number = 5, 0
        for i in range(len(Y)):
            if ids is not None and len(ids) == len(Y):
                worksheet.write(row_number, col_number, str(ids[i]))
            for t in range(len(X)):
                col_number += 1
                worksheet.write(row_number, col_number, Y[i][t])
            col_number = 0
            row_number += 1
        workbook.close()
        print("excel file saved")

    save_data_file = save_data_file3

else:
    def save_data_file2(X, Y, X_unit, name, nos, keyword, func, use_name, ids=None):
        docName = func + "_" + nos + "_" + keyword + "_" + use_name

        # Save Neuron List as Text
        Ymod = [None] * len(Y)
        docNameTxt = "NeuronLists_" + docName + ".txt"
        for i in range(len(Y)):
            Ymod[i] = nos + str(ids[i]) + "=" + str(Y[i])
        with open(docNameTxt, 'w+') as fileHandle:
            for y in Ymod:
                fileHandle.write('%s\n' % y)
        with open(docNameTxt, "a") as myfile:
            myfile.write("\n" + nos + " values for time over " + str(len(X)) + X_unit + ".")

        # Save text file by time
        docNameTime = "TimeSeparated_" + docName + ".csv"
        new_list = zip(Y, X)
        with open(docNameTime, 'w+') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerows(new_list)

        # Save as Dataframe
        secondsVals = []
        nosList = []

        csvDocName = "CSV_" + docName + ".csv"
        xu = 0
        yu = 0
        while xu < len(X):
            secondsVals.append(str(xu) + X_unit)
            xu += 1
        while yu < len(Y):
            nosList.append(nos + "" + str(yu))
            yu += 1
        df = pd.DataFrame(Y, columns=secondsVals, index=nosList)
        df.to_csv(csvDocName)

    save_data_file = save_data_file2


def apply_func(app_func, value_list, times):
    if app_func == "raw":
        return times, [values_in[times] for values_in in value_list], "ms"
    elif app_func == "mean":
        value_list = [np.expand_dims(values_in, 1) for values_in in [values_in[times] for values_in in value_list]]
        value_arr = np.concatenate(value_list, axis=1)
        value_arr = np.mean(value_arr, axis=1)
        return times, [value_arr], "ms"
    elif app_func == "psd":
        value_list = [np.expand_dims(values_in, 1) for values_in in [values_in[times] for values_in in value_list]]
        value_arr = np.concatenate(value_list, axis=1)
        ff = np.power(np.abs(np.fft.fft(value_arr, axis=0)), 2)
        omega = np.fft.fftfreq(len(times), timestep/1000)
        ff = (ff[:ff.shape[0] // 2, :]).transpose()
        omega = omega[:omega.shape[0] // 2]
        nd = np.argmin(np.abs(omega - 150))
        return omega[:nd], ff[:, :nd], "Hz"
    elif app_func == "mean_psd":
        value_list = [np.expand_dims(values_in, 1) for values_in in [values_in[times] for values_in in value_list]]
        value_arr = np.concatenate(value_list, axis=1)
        ff = np.power(np.abs(np.fft.fft(value_arr, axis=0)), 2)
        omega = np.fft.fftfreq(len(times), timestep / 1000)
        ff = (ff[:ff.shape[0] // 2, :])
        omega = omega[:omega.shape[0] // 2]
        ff = np.mean(ff, axis=1).transpose()
        nd = np.argmin(np.abs(omega - 150))
        return omega[:nd], [ff[:nd]], "Hz"


def simplify_plot(X, Y, X_unit, name):
    rows = int(np.sqrt(len(Y)))
    cols = int(np.ceil(len(Y)/rows))
    f, ax = plt.subplots(rows, cols, sharex='all', sharey='all')
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= len(Y):
                break
            if rows*cols == 1:
                ax.plot(X, Y[count])
            else:
                ax[r, c].plot(X, Y[count])
            count += 1
        if count >= len(Y):
            break
    plt.suptitle(name+" with x-axis unit="+X_unit)



def generate_save_or_plot(line=-1):
    try:
        ttu = Interpreter.get_save_selection(line)
    except KeyError:
        print("Error: there was an issue with the given input. (A neuron/synapse group name was probably wrong).")
        ttu = None
    rv = 1
    if ttu == "end":
        return 0
    elif ttu is None:
        pass
    else:
        sop, nos, val_type, func, object_list, times, use_name = ttu
        if func == "raster" and sop == "plot" and nos == "neuron":
            sub_raster(object_list, times)
            return 2
        elif func == "raster":
            print("incorrect use of raster (must be plot and neuron), please consult manual")
        else:
            try:
                val_lst = [x.printable_dict[val_type] for x in object_list]
                if func == "spectrogram" or func == "spec" and sop == "plot":
                    plt_subspec(val_lst, times, use_name)
                    return 2
                elif func == "spectrogram" or func == "spec":
                    print("Error: spectrogram can only be plotted")
                else:
                    rt = apply_func(func, val_lst, times)
                    if rt is None:
                        print("invalid func: recommend trying \"raw\" or \"mean\"")
                        return 1
                    X, Y, X_unit = rt
                    title = "" + func + " of " + nos + "s " + val_type + " for " + use_name
                    if sop == "plot":
                        simplify_plot(X, Y, X_unit, title)
                        return 2
                    elif sop == "save":
                        if nos == "neuron":
                            ids = [n.id for n in object_list]
                        else:
                            ids = [(s.pre.id, s.post.id) for s in object_list]
                        title = "" + func + " of " + nos + "s " + val_type
                        print("saving...")
                        save_data_file(X, Y, X_unit, title, nos, val_type, func, use_name, ids=ids)

            except KeyError:
                print("invalid keyword")
    return rv


def significance(neu, t1, t2, t3, t4):
    n1 = t2-t1
    n2 = t4-t3
    accu1 = np.zeros((len(neuron_dict[neu]),n1))
    accu2 = np.zeros((len(neuron_dict[neu]),n2))
    for i in range(len(neuron_dict[neu])):
        accu1[i,:] = neuron_dict[neu][i].output_history[t1:t2]
        accu2[i,:] = neuron_dict[neu][i].output_history[t3:t4]
    stat, p = stts.ttest_rel(accu1.flatten(), accu2.flatten())
    print("sample 1 mean: " + str(accu1.flatten().mean()) + " standard deviation: " +
          str(round(accu1.flatten().std(),5)))
    print("sample 2 mean: " + str(accu2.flatten().mean()) + " standard deviation: " +
          str(round(accu2.flatten().std(),5)))
    print("stat: " + str(round(stat,3)) + " p-value: " + str(round(p,5)))


def plot_difference(neu, t1, t2, t3, t4):
    assert t2-t1 == t4-t3
    n = t4-t3
    accu1 = np.zeros(len(neuron_dict[neu]))
    accu2 = np.zeros(len(neuron_dict[neu]))
    for i in range(len(neuron_dict[neu])):
        accu1[i] = neuron_dict[neu][i].output_history[t1:t2].sum()
        accu2[i] = neuron_dict[neu][i].output_history[t3:t4].sum()
    diff = accu2-accu1
    #stats
    print("Mean of differences: " + str(round(diff.mean(),4)))
    print("Standard deviation of differences: " + str(round(diff.std(),4)))
    #scatter plot distribution
    fig, axs = plt.subplots(1, 1)
    axs.plot(np.arange(len(accu1)),accu1,label="firings from " + str(t1) + " to " + str(t2))
    axs.plot(np.arange(len(accu1)),accu2,label="firings from " + str(t3) + " to " + str(t4))
    plt.xlabel("neuron number")
    plt.ylabel("count")
    plt.legend()
    #histograms
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(diff,label="differences in firings (count)")
    axs[1].plot(np.arange(len(accu1)),diff,label="differences in firings")
    fig.legend()
    #accending
    diff.sort()
    fig, axs = plt.subplots(1, 1)
    axs.plot(np.arange(len(accu1)),diff,label="differences in firings in ascending order")
    plt.ylabel("count")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    seed = 2
    pn = 3

    # get model file name from metadata.txt or from console
    file = Interpreter.get_filename()

    # building or loading model
    vals = Interpreter.interpret_file(file)
    timestep, _, total_neuron_number, all_neurons, neuron_dict, group_names, all_connections, \
        connection_dict, connection_names, external_inputs, sniff_frequency, nclass_dict, save_info\
        = vals
    load_dir, keep_load, save_dir = save_info

    starting_timestep, final_timestep = saver.load_data(vals)

    for neuron in all_neurons:
        neuron.update_starting_parameters(load_dir == "")
    for synapse in all_connections:
        synapse.update_starting_parameters(load_dir == "")

    # running model
    print("setup complete, beginning run")
    time_taken = []
    for t in range(starting_timestep, final_timestep):

        s = time.time()
        for neuron in all_neurons:
            if (neuron.id, t) in external_inputs:
                # if neuron.id == 75:
                #     print(t, external_inputs[(neuron.id, t)])
                neuron.update_voltage(t, timestep, external_inputs[(neuron.id, t)])
            else:
                neuron.update_voltage(t, timestep, 0)
        for synapse in all_connections:
            synapse.update_weights(t, timestep)
        time_taken.append(time.time() - s)
        print("step number:", t,
              "\tpercent done:",
              np.round((100.0 * (t - starting_timestep + 1)) / (final_timestep - starting_timestep + 1), 2),
              "\ttime per step estimate:",
              np.round(np.mean(time_taken[max(0, len(time_taken) - 10):len(time_taken)]), 3),
              "\ttime estimate:",
              np.round(np.mean(time_taken[max(0, len(time_taken) - 50):len(time_taken)]) * (final_timestep - t), 1))

    # make save
    saver.save(vals)
    print("he", all_neurons[0].voltage_history.shape)
    # make predefined data plots or saves
    not_first_plot = False
    print(Interpreter.predef_output_lines)
    ret = 1
    any_plots = False
    for line in Interpreter.predef_output_lines:
        ret = generate_save_or_plot(line)
        if ret == 0:
            break
        if ret == 2:
            not_first_plot = True
            any_plots = True

    if any_plots:
        print("You must close the plots to progress to user inputs")
        plt.show()

    # make user defined plots or saves
    while ret:
        ret = generate_save_or_plot()
        if ret == 2:
            not_first_plot = True
            print("You must close the plot to continue to use user inputs")
            plt.show()

    stats = 1
    print("Test for statistics in Voltage and Firing Rate")
    while stats:
        try:
            sinp = Interpreter.input_fun("type (difference) or (end): ")
            if sinp == "end":
                break
            if sinp != "difference" and sinp != "firing":
                raise Exception()
            neu = Interpreter.input_fun("neuron type: ")
            t1 = int(Interpreter.input_fun("start time for first sample: "))
            t2 = int(Interpreter.input_fun("end time for first sample: "))
            t3 = int(Interpreter.input_fun("start time for second sample: "))
            t4 = int(Interpreter.input_fun("end time for second sample: "))
            if sinp == "firing":
                significance(neu,t1,t2,t3,t4)
            else:
                plot_difference(neu,t1,t2,t3,t4)
        except:
            print("invalid input")

    print("Thank you")
