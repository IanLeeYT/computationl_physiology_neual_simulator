import numpy as np


class Neuron:

    def __init__(self, id_no, final_timestep):
        self.id = id_no
        self.param_dict = {
            "time_constant": 1.,
            "resting_voltage": -70.,
            "upper_threshold": -55.,
            "lower_threshold": -65.,
            "axon_length": 0,
            "refractory_voltage": 0,
            "sigmoid_slope": 0,
            "sigmoid_center": 0,

        }
        self.pre = []
        self.post = []
        self.voltage_history = np.zeros(final_timestep)
        self.output_history = np.zeros(final_timestep)

    def update_starting_parameters(self, not_loaded):
        """called once, after all base parameters are set, to update other dependant values"""
        if not_loaded:
            self.voltage_history[0] = self.param_dict["resting_voltage"]

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
            "resting_voltage": -70.,
        }
        self.pre = []
        self.post = []
        self.voltage_history = np.zeros(final_timestep)
        self.output_history = np.zeros(final_timestep)

    def update_starting_parameters(self):
        """called once, after all base parameters are set, to update other dependant values"""
        self.voltage_history[0] = self.param_dict["resting_voltage"]

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
            self.param_dict["resting_voltage"]
        v_ext = 0
        for p in self.pre:
            v_ext += p.get_input(step_number, timestep)
        exponent = np.exp(-1 * timestep / self.param_dict["time_constant"])
        self.voltage_history[step_number] = exponent * prev_v + (1 - exponent) * v_ext + artificial_stimulus


class CPNeuron:

    def __init__(self, id_no, final_timestep):
        # TODO: INCLUDE ANY PARAMETERS NECESSARY FOR CALCIUM ADAPTATION
        self.id = id_no
        self.param_dict = {
            "time_constant": 1.,
            "resting_voltage": -70.,
            "upper_threshold": -55.,
            "lower_threshold": -65.,
            "axon_length": 0,
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

    def update_starting_parameters(self, not_loaded):
        """called once, after all base parameters are set, to update other dependant values"""
        if not_loaded:
            self.voltage_history[0] = self.param_dict["resting_voltage"]

    def update_voltage(self, step_number, timestep, artificial_stimulus):
        # TODO: INCLUDE CALCIUM ADAPTATION SO THAT THE NEURON IS LESS LIKELY TO FIRE WITH A CONSTANT INPUT.
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

neuron_names = {
    "basic": Neuron,
    "nsn": NSNeuron,
    "calcium_pyr": CPNeuron,
}
