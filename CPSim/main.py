import Interpreter
import numpy as np
import matplotlib.pyplot as plt
import time
import saver
import threading
import xlsxwriter as xw


def raster():
    # plots a raster plot (spike timing plot) for all neurons. colors different groups different colors
    fires = [np.greater(np.array(all_neurons[i].output_history), 0.1) for i in range(total_neuron_number)]
    fire_times = [[j for j in range(len(fires[i])) if fires[i][j]] for i in range(total_neuron_number)]
    fig, axs = plt.subplots(1, 1)
    axs.eventplot(fire_times, linelengths=0.5, colors="black")
    plt.ylabel("Neuron number")
    plt.xlabel("timestep")
    plt.title("raster")


def save_voltage_data_file(neurons, indices, times, name="Neuron_Data"):
    """
    neuron: list of neuron types desired
    indices: tuple/list of tuples that indicates the starting and ending neurons
    times: tuple of start and end times of recording

    writes data into an excel file
    """
    assert isinstance(indices, (list, tuple)), "invalid input: must be tuple or list of tuples"
    assert isinstance(times, tuple), "invalid time input: must be tuple"
    if isinstance(indices, tuple):
        indices = [indices] * len(neurons)

    workbook = xw.Workbook(name + ".xlsx")
    worksheet = workbook.add_worksheet()

    row_number, col_number = 0, 0
    title = "The following data records voltages of neuron groups " + str(neurons) \
            + " from time " + str(times[0]) + " to " + str(times[1]) + " seconds"
    comment1 = "Row headers are neurons and their indices."
    comment2 = "Column headers are times in seconds"
    worksheet.write(row_number, col_number, title)
    row_number += 1
    worksheet.write(row_number, col_number, comment1)
    row_number += 1
    worksheet.write(row_number, col_number, comment2)

    row_number, col_number = 4, 1
    time_array = np.arange((times[1] - times[0]) / timestep)
    for t in range(len(time_array)):
        worksheet.write(row_number, col_number, t * timestep + times[0])
        col_number += 1

    row_number, col_number = 5, 0
    for n in range(len(neurons)):
        for i in range(indices[n][0], indices[n][1] + 1):
            worksheet.write(row_number, col_number, neurons[n] + " " + str(i))
            for t in range(len(time_array)):
                col_number += 1
                neuron = neuron_dict[neurons[n]][i]
                worksheet.write(row_number, col_number, neuron.voltage_history[t])
            col_number = 0
            row_number += 1
    workbook.close()
    print("excel file saved")


def post_processing_inputs():
    # TODO
    pass


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
    print("external inputs", external_inputs)
    print(timestep)

    for neuron in all_neurons:
        neuron.update_starting_parameters(load_dir == "")
    for synapse in all_connections:
        synapse.update_starting_parameters(load_dir == "")

    # running model
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

    # make plots
    nome = ["OSN", "Mi", "PG", "GC", "ET", "Pyr"]
    nome2 = ["OSN", "Mi", "GC", "Pyr"]
    use = nome
    plt.title("voltages and spikes for a column")
    for i in range(len(group_names)):
        plt.subplot(320 + i + 1)
        for x in range(final_timestep):
            if all_neurons[100 * i + 50].output_history[x]:
                plt.axvline(x, color="lightpink", linestyle='dashed')
        plt.plot(all_neurons[100 * i + 50].voltage_history)
        plt.title(use[i])
    plt.tight_layout(pad=1)


    # plot weight changes
    plt.figure()
    nome3 = ["OSN_to_PG", "OSN_to_Mi", "OSN_to_ET", "Mi_to_GC", "GC_to_Mi", "Mi_to_Pyr", "Pyr_to_Pyr"]
    plt.title("weight change history for syanpses")
    for i in range(len(connection_names)):
        plt.subplot(420 + i + 1)
        plt.title(nome3[i])
        for x in range(final_timestep):
            if connection_dict[nome3[i]][len(connection_dict[nome3[i]])//2].pre.output_history[x]:
                plt.axvline(x, color="lightpink", linestyle='dashed')
            if connection_dict[nome3[i]][len(connection_dict[nome3[i]])//2].post.output_history[x]:
                plt.axvline(x, color="lightgreen", linestyle='dashed')
        if i == 6:
            v = np.zeros(final_timestep)
            for n in connection_dict[nome3[i]]:
                v += n.weight_history
            plt.plot(v/len(connection_dict[nome3[i]]))
        else:
            plt.plot(connection_dict[nome3[i]][len(connection_dict[nome3[i]])//2].weight_history)
    plt.tight_layout(pad=1)


    # plotting power spectra
    plt.figure()
    vs = np.zeros((total_neuron_number, final_timestep))
    for j in range(total_neuron_number):
        vs[j] = np.copy(all_neurons[j].output_history)
    # calculates the power spectra below
    ff = np.power(np.abs(np.fft.fft(vs, axis=1)), 2)
    for i in range(len(use)):
        plt.subplot(320 + i + 1)
        # does the averaging of each group below
        m = np.mean(ff[i * 100: i * 100 + 100], axis=0)
        plt.plot(np.fft.fftfreq(final_timestep, 0.001)[:m.shape[0] // 2], m[:m.shape[0] // 2])
        plt.title(use[i] + "fft")
        plt.xlim((0, 100))
        plt.ylim((0, 4000))
    plt.tight_layout(pad=1)

    # plotting spectrogram
    plt.figure()
    plt.title("spectrogram")
    Fs = 1000  # sampling frequency
    for i in range(len(group_names)):
        plt.subplot(320 + i + 1)
        plt.title(use[i] + " spec")
        v = np.zeros(final_timestep)
        for n in all_neurons[i*(100):i*(100)+100]:
            v += n.output_history

        # f, t, Sxx = signal.spectrogram(x/100, fs = 1000)
        # plt.pcolormesh(t, f, Sxx, shading='gouraud')
        nfft = 100
        noverlap = 85
        if i == 6:
            nfft = 200
        plt.specgram(v/100, Fs=Fs, NFFT=nfft, noverlap=noverlap)
        plt.xlabel('Time[s]')
        plt.ylabel('Frequency[Hz]')
        plt.ylim((0, 100))
    plt.tight_layout(pad=1.5)

    print("total time taken:", sum(time_taken))
    raster()
    # plt.show(block=False)
    plt_thread = threading.Thread(target=post_processing_inputs, daemon=True)
    plt_thread.start()

    save_voltage_data_file(["OSN", "Mitral", "PG", "GC", "Pyr"], (0, 99), (0, 200), "Neuron_Data")

    plt.show()
