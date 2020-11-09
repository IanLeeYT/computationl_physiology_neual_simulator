import numpy as np
import Cache


class Neuron:

    def __init__(self, id_no, final_timestep):
        self.id = id_no
        self.param_dict = {
            "time_constant": 1.,
            "refractory_voltage": 0,
            "sigmoid_slope": 0,
            "sigmoid_center": 0,
        }
        self.pre = []
        self.post = []
        self.voltage_history = np.zeros(final_timestep)
        self.output_history = np.zeros(final_timestep)
        self.printable_dict = {
            "voltage": self.voltage_history,
            "output": self.output_history
        }

    def update_starting_parameters(self, not_loaded):
        """called once, after all base parameters are set, to update other dependant values"""
        pass

    def update_voltage(self, step_number, timestep, artificial_stimulus):
        """
        Called once for every neuron at every timestep.

        must set self.voltage_history[step_number]
                 self.output_history[step_number]
                 self.v_ext_history[step_number] (optional, used for debugging only)

        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        :param artificial_stimulus: artificial external input (usually to represent odor receptor response)
        """
        prev_step_number = step_number - 1
        prev_v = self.voltage_history[prev_step_number] if self.output_history[prev_step_number] == 0 else\
            self.param_dict["refractory_voltage"]
        v_ext = 0
        for p in self.pre:
            v_ext += p.get_input(step_number, timestep)
        exponent = np.exp(-1 * timestep / self.param_dict["time_constant"])
        self.voltage_history[step_number] = exponent * prev_v + (1 - exponent) * v_ext + artificial_stimulus
        p_val = np.random.random()
        comp = self.param_dict["sigmoid_slope"]
        if comp != 0:
            sc = 1 / (1 + np.exp(-1 * (self.voltage_history[step_number] - self.param_dict["sigmoid_center"]) / comp))

            self.output_history[step_number] = sc > p_val
        else:
            self.output_history[step_number] = self.voltage_history[step_number] > self.param_dict["sigmoid_center"]


class NSNeuron:

    def __init__(self, id_no, final_timestep):
        self.id = id_no
        self.param_dict = {
            "time_constant": 1.,
        }
        self.pre = []
        self.post = []
        self.voltage_history = np.zeros(final_timestep)
        self.output_history = np.zeros(final_timestep)
        self.printable_dict = {
            "voltage": self.voltage_history,
            "output": self.output_history
        }

    def update_starting_parameters(self):
        """called once, after all base parameters are set, to update other dependant values"""
        pass

    def update_voltage(self, step_number, timestep, artificial_stimulus):
        """
        Called once for every neuron at every timestep.

        must set self.voltage_history[step_number]
                 self.output_history[step_number]
                 self.v_ext_history[step_number] (optional, used for debugging only)

        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        :param artificial_stimulus: artificial external input (usually to represent odor receptor response)
        """
        prev_step_number = step_number - 1
        prev_v = self.voltage_history[prev_step_number] if self.output_history[prev_step_number] == 0 else\
            0
        v_ext = 0
        for p in self.pre:
            v_ext += p.get_input(step_number, timestep)
        exponent = np.exp(-1 * timestep / self.param_dict["time_constant"])
        self.voltage_history[step_number] = exponent * prev_v + (1 - exponent) * v_ext + artificial_stimulus


class CPNeuron:

    def __init__(self, id_no, final_timestep):
        self.id = id_no
        self.param_dict = {
            "time_constant": 1.,
            "refractory_voltage": 0,
            "sigmoid_slope": 0,
            "sigmoid_center": 0,
            "calcium_inhibition_sample_t": 1000,
            "calcium_inhibition_slope": 25
        }
        self.pre = []
        self.post = []
        self.voltage_history = np.zeros(final_timestep)
        self.output_history = np.zeros(final_timestep)
        self.printable_dict = {
            "voltage": self.voltage_history,
            "output": self.output_history
        }

    def update_starting_parameters(self, not_loaded):
        """called once, after all base parameters are set, to update other dependant values"""
        pass

    def update_voltage(self, step_number, timestep, artificial_stimulus):
        """
        Called once for every neuron at every timestep.

        must set self.voltage_history[step_number]
                 self.output_history[step_number]
                 self.v_ext_history[step_number] (optional, used for debugging only)

        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        :param artificial_stimulus: artificial external input (usually to represent odor receptor response)
        """
        """
        update voltage function:
        based on previous 100 neurons, the sigmoid center is adjusted
        """
        prev_step_number = step_number - 1
        prev_v = self.voltage_history[prev_step_number] if self.output_history[prev_step_number] == 0 else\
            self.param_dict["refractory_voltage"]
        v_ext = 0
        for p in self.pre:
            v_ext += p.get_input(step_number, timestep)
        exponent = np.exp(-1 * timestep / self.param_dict["time_constant"])
        self.voltage_history[step_number] = exponent * prev_v + (1 - exponent) * v_ext + artificial_stimulus
        p_val = np.random.random()
        comp = self.param_dict["sigmoid_slope"]
        adjusted_sigmoid = self.calcium_inhibition_value(step_number, timestep) + self.param_dict["sigmoid_center"]

        if comp != 0:
            sc = 1 / (1 + np.exp(-1 * (self.voltage_history[step_number] - adjusted_sigmoid) / comp))
            self.output_history[step_number] = sc > p_val
        else:
            self.output_history[step_number] = self.voltage_history[step_number] > adjusted_sigmoid

    def calcium_inhibition_value(self, step_number, timestep):
        """
        Return the increase in sigmoid center due to prolonged exposure to constant input

        Calculated based on number of previous firings in a time interval
        """
        if step_number * timestep <= self.param_dict["calcium_inhibition_sample_t"]:
            k = np.sum(self.output_history[:step_number])
        else:
            k = np.sum(self.output_history[int(step_number - self.param_dict["calcium_inhibition_sample_t"]/timestep):
                                           step_number])
        coef = np.exp(k / self.param_dict["calcium_inhibition_slope"]) - 1
        # if k != 0:
        #     print("cal", k, coef)
        return coef


class CP2Neuron:

    def __init__(self, id_no, final_timestep):
        self.id = id_no
        self.param_dict = {
            "time_constant": 1.,
            "refractory_voltage": 0,
            "sigmoid_slope": 0,
            "sigmoid_center": 0,

            "ca_inhib_relevant_spike_no": 20,
            "ca_inhib_tau1": 10.,
            "ca_inhib_tau2": 20.,
            "ca_inhib_weight": 0.1,
            "ca_inhib_E": -10
        }
        self.pre = []
        self.post = []
        self.voltage_history = np.zeros(final_timestep)
        self.output_history = np.zeros(final_timestep)
        self.ca_inhib_gmax = 0;
        self.printable_dict = {
            "voltage": self.voltage_history,
            "output": self.output_history
        }

    def update_starting_parameters(self, not_loaded):
        """called once, after all base parameters are set, to update other dependant values"""

        t1 = self.param_dict["ca_inhib_tau1"]
        t2 = self.param_dict["ca_inhib_tau2"]
        if t1 == t2:
            self.ca_inhib_gmax = 0
        elif t1 == 0:
            self.ca_inhib_gmax = -1
        elif t2 == 0:
            self.ca_inhib_gmax = 1
        else:
            mx = t1 * t2 * (np.log(t2) - np.log(t1)) / (t2 - t1)
            self.ca_inhib_gmax = 1 / (np.exp(-1 * mx / t1) - np.exp(-1 * mx / t2))

    def get_ca_self_inhib(self, step_number, timestep):
        prev_v = self.voltage_history[step_number-1] if self.output_history[step_number-1] == 0 else\
            self.param_dict["refractory_voltage"]
        key_g_sum = (step_number, ("sum_g", self.id, self.param_dict["ca_inhib_relevant_spike_no"],
                                   self.param_dict["ca_inhib_tau1"], self.param_dict["ca_inhib_tau2"]))
        key_rsi = (step_number, ("rsi", self.id, self.param_dict["ca_inhib_relevant_spike_no"]))
        if Cache.cache.search(key_g_sum):
            sum_g = Cache.cache.get(key_g_sum)
        else:
            if Cache.cache.search(key_rsi):
                rsi = Cache.cache.get(key_rsi)
            else:
                rs = np.arange(self.output_history.shape[0])[max(0, step_number - 100): step_number] \
                    [np.greater(self.output_history, 0)[max(0, step_number - 100): step_number]]
                if rs.shape[0] > self.param_dict["ca_inhib_relevant_spike_no"]:
                    rs = rs[-1 * int(self.param_dict["ca_inhib_relevant_spike_no"]):]
                rsi = -1 * timestep * (step_number - rs)
                Cache.cache.store(key_rsi, rsi)

            g = self.ca_inhib_gmax * (np.exp(rsi / self.param_dict["ca_inhib_tau1"]) -
                                      np.exp(rsi / self.param_dict["ca_inhib_tau2"]))
            sum_g = np.sum(g)
            Cache.cache.store(key_g_sum, sum_g)

        cur = self.param_dict["ca_inhib_weight"] * sum_g * (self.param_dict["ca_inhib_E"] - prev_v)
        return cur

    def update_voltage(self, step_number, timestep, artificial_stimulus):
        """
        Called once for every neuron at every timestep.

        must set self.voltage_history[step_number]
                 self.output_history[step_number]
                 self.v_ext_history[step_number] (optional, used for debugging only)

        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        :param artificial_stimulus: artificial external input (usually to represent odor receptor response)
        """
        """
        update voltage function:
        based on previous 100 neurons, the sigmoid center is adjusted
        """
        prev_step_number = step_number - 1
        prev_v = self.voltage_history[prev_step_number] if self.output_history[prev_step_number] == 0 else\
            self.param_dict["refractory_voltage"]
        v_ext = 0
        for p in self.pre:
            v_ext += p.get_input(step_number, timestep)

        v_ext += self.get_ca_self_inhib(step_number, timestep)

        exponent = np.exp(-1 * timestep / self.param_dict["time_constant"])
        self.voltage_history[step_number] = exponent * prev_v + (1 - exponent) * v_ext + artificial_stimulus
        p_val = np.random.random()
        comp = self.param_dict["sigmoid_slope"]
        if comp != 0:
            sc = 1 / (1 + np.exp(-1 * (self.voltage_history[step_number] - self.param_dict["sigmoid_center"]) / comp))

            self.output_history[step_number] = sc > p_val
        else:
            self.output_history[step_number] = self.voltage_history[step_number] > self.param_dict["sigmoid_center"]


neuron_names = {
    "basic": Neuron,
    "nsn": NSNeuron,
    "calcium_pyr": CPNeuron,
    "calcium_pyr_inhib": CP2Neuron
}
