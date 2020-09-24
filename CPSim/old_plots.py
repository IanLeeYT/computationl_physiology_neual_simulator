# make plots
# nome = ["OSN", "Mi", "PG", "GC", "ET", "Pyr"]
# nome2 = ["OSN", "Mi", "GC", "Pyr"]
# use = nome
# plt.title("voltages and spikes for a column")
# for i in range(len(group_names)):
#     plt.subplot(320 + i + 1)
#     for x in range(final_timestep):
#         if all_neurons[100 * i + 50].output_history[x]:
#             plt.axvline(x, color="lightpink", linestyle='dashed')
#     plt.plot(all_neurons[100 * i + 50].voltage_history)
#     plt.title(use[i])
# plt.tight_layout(pad=1)


# plot weight changes
# plt.figure()
# nome3 = ["OSN_to_PG", "OSN_to_Mi", "OSN_to_ET", "Mi_to_GC", "GC_to_Mi", "Mi_to_Pyr", "Pyr_to_Pyr"]
# plt.title("weight change history for syanpses")
# for i in range(len(connection_names)):
#     plt.subplot(420 + i + 1)
#     plt.title(nome3[i])
#     for x in range(final_timestep):
#         if connection_dict[nome3[i]][len(connection_dict[nome3[i]])//2].pre.output_history[x]:
#             plt.axvline(x, color="lightpink", linestyle='dashed')
#         if connection_dict[nome3[i]][len(connection_dict[nome3[i]])//2].post.output_history[x]:
#             plt.axvline(x, color="lightgreen", linestyle='dashed')
#     if i == 6:
#         v = np.zeros(final_timestep)
#         for n in connection_dict[nome3[i]]:
#             v += n.weight_history
#         plt.plot(v/len(connection_dict[nome3[i]]))
#     else:
#         plt.plot(connection_dict[nome3[i]][len(connection_dict[nome3[i]])//2].weight_history)
# plt.tight_layout(pad=1)
#
#
# # plotting power spectra
# plt.figure()
# vs = np.zeros((total_neuron_number, final_timestep))
# for j in range(total_neuron_number):
#     vs[j] = np.copy(all_neurons[j].output_history)
# # calculates the power spectra below
# ff = np.power(np.abs(np.fft.fft(vs, axis=1)), 2)
# for i in range(len(use)):
#     plt.subplot(320 + i + 1)
#     # does the averaging of each group below
#     m = np.mean(ff[i * 100: i * 100 + 100], axis=0)
#     plt.plot(np.fft.fftfreq(final_timestep, 0.001)[:m.shape[0] // 2], m[:m.shape[0] // 2])
#     plt.title(use[i] + "fft")
#     plt.xlim((0, 100))
#     plt.ylim((0, 4000))
# plt.tight_layout(pad=1)
#
# # plotting spectrogram
# plt.figure()
# plt.title("spectrogram")
# Fs = 1000  # sampling frequency
# for i in range(len(group_names)):
#     plt.subplot(320 + i + 1)
#     plt.title(use[i] + " spec")
#     v = np.zeros(final_timestep)
#     for n in all_neurons[i*(100):i*(100)+100]:
#         v += n.output_history
#
#     # f, t, Sxx = signal.spectrogram(x/100, fs = 1000)
#     # plt.pcolormesh(t, f, Sxx, shading='gouraud')
#     nfft = 100
#     noverlap = 85
#     if i == 6:
#         nfft = 200
#     plt.specgram(v/100, Fs=Fs, NFFT=nfft, noverlap=noverlap)
#     plt.xlabel('Time[s]')
#     plt.ylabel('Frequency[Hz]')
#     plt.ylim((0, 100))
# plt.tight_layout(pad=1.5)
#
# print("total time taken:", sum(time_taken))
# raster()
# # plt.show(block=False)
# plt_thread = threading.Thread(target=post_processing_inputs, daemon=True)
# plt_thread.start()
# plt.show()
# save_voltage_data_file(["OSN", "Mitral", "PG", "GC", "Pyr"], (0, 99), (0, 200), "Neuron_Data")
