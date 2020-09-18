import numpy as np
import random


class Synapse:
    def __init__(self, f, t, final_timestep):
        self.pre = f
        self.post = t
        self.pre.post.append(self)
        self.post.pre.append(self)
        self.param_dict = {
            "E": -70.,
            "tau1": 1.,
            "tau2": 2.,
            "initial_weight": 1.,
            "weight_change": 0.0,
            "gmax": 1.,
            "relevant_period": 0,
            "decay": 0.05,
            "relevant_spike_no": 20,
            "sniff_cycle_period": 100
        }
        self.weight_history = np.zeros(final_timestep)
        self.g_max = 1

    def __repr__(self):
        return "" + str(self.pre.id) + " " + str(self.post.id)

    def update_starting_parameters(self, not_loaded):
        if not_loaded:
            self.weight_history[0] = self.param_dict["initial_weight"]
        t1 = self.param_dict["tau1"]
        t2 = self.param_dict["tau2"]
        if t1 == t2:
            self.g_max = 0
        elif t1 == 0:
            self.g_max = -1
        elif t2 == 0:
            self.g_max = 1
        else:
            mx = t1 * t2 * (np.log(t2) - np.log(t1)) / (t2 - t1)
            self.g_max = 1 / (np.exp(-1 * mx / t1) - np.exp(-1 * mx / t2))

    def get_input(self,  step_number, timestep):
        """
        Calculate and return the output of this synapse
        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        :return: The effect this synapse will have on its post neuron
        """
        rs = np.arange(self.pre.output_history.shape[0])[max(0, step_number - 100): step_number]\
                    [np.greater(self.pre.output_history, 0)[max(0, step_number - 100): step_number]]
        if rs.shape[0] > self.param_dict["relevant_spike_no"]:
            rs = rs[-1 * int(self.param_dict["relevant_spike_no"]):]
        rsi = -1 * timestep * (step_number - rs)

        g = self.g_max * (np.exp(rsi / self.param_dict["tau1"]) - np.exp(rsi / self.param_dict["tau2"]))
        # g = np.abs(1 * (np.exp(rsi / self.param_dict["tau1"]) - np.exp(rsi / self.param_dict["tau2"])))

        prev_v = self.post.voltage_history[step_number - 1] if self.post.output_history[step_number - 1] == 0 else \
            self.post.param_dict["resting_voltage"]

        cur = self.weight_history[step_number - 1] * np.sum(g) * (self.param_dict["E"] - prev_v)
        # if(self.post.id == 350):
        #     print(g, cur)

        return cur

    def update_weights(self,  step_number, timestep):
        """
        Use a learning rule to update the weight of this synapse
        must set self.weight_history[step_number]

        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        """
        prev_w = ((self.weight_history[step_number - 1] - self.param_dict["initial_weight"]) *
                  np.exp(-1 * timestep / self.param_dict["decay"])) + self.param_dict["initial_weight"]
        if self.param_dict["weight_change"] != 0.:
            tp = self.param_dict["sniff_cycle_period"]
            if step_number >= tp:
                st = ((step_number - tp) // tp) * tp
                nd = min(st + tp, len(self.weight_history))
                x1 = np.sum(self.pre.output_history[st: nd])
                x2 = np.sum(self.post.output_history[st: nd])
            else:
                x2 = x1 = 0
            del_w = self.param_dict["weight_change"] * x1 * x2
            self.weight_history[step_number] = prev_w + del_w
        else:
            self.weight_history[step_number] = prev_w


class NonfireSynapse:

    def __init__(self, f, t, final_timestep):
        self.pre = f
        self.post = t
        self.pre.post.append(self)
        self.post.pre.append(self)
        self.param_dict = {
            "s_sigmoid_slope": 1.,
            "s_sigmoid_center": 0.,
            "initial_weight": 1.,
            "weight_change": 0.0,
            "decay": 0.05,
            "sniff_cycle_period": 100
        }
        self.weight_history = np.zeros(final_timestep)

    def __repr__(self):
        return "" + str(self.pre.id) + " " + str(self.post.id)

    def update_starting_parameters(self, not_loaded):
        if not_loaded:
            self.weight_history[0] = self.param_dict["initial_weight"]
        t1 = self.param_dict["tau1"]
        t2 = self.param_dict["tau2"]
        if t1 == t2:
            self.g_max = 0
        elif t1 == 0:
            self.g_max = -1
        elif t2 == 0:
            self.g_max = 1
        else:
            mx = t1 * t2 * (np.log(t2) - np.log(t1)) / (t2 - t1)
            self.g_max = 1 / (np.exp(-1 * mx / t1) - np.exp(-1 * mx / t2))

    def get_input(self,  step_number, timestep):
        """
        Calculate and return the output of this synapse
        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        :return: The effect this synapse will have on its post neuron
        """
        comp = self.param_dict["s_sigmoid_slope"]
        cent = self.param_dict["s_sigmoid_center"]

        cur = self.weight_history[step_number - 1] / \
            (1 + np.exp(-1 * (self.pre.voltage_history[step_number-1] - cent) / comp))
        if self.pre.id % 100 == 50:
            print(self.pre.id, self.post.id, self.pre.voltage_history[step_number-1], cur)
        return cur

    def update_weights(self,  step_number, timestep):
        """
        Use a learning rule to update the weight of this synapse
        must set self.weight_history[step_number]

        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        """
        prev_w = ((self.weight_history[step_number - 1] - self.param_dict["initial_weight"]) *
                  np.exp(-1 * timestep / self.param_dict["decay"])) + self.param_dict["initial_weight"]
        if self.param_dict["weight_change"] != 0.:
            tp = self.param_dict["sniff_cycle_period"]
            if step_number >= tp:
                st = ((step_number - tp) // tp) * tp
                nd = min(st + tp, len(self.weight_history))
                x1 = np.sum(self.pre.output_history[st: nd])
                x2 = np.sum(self.post.output_history[st: nd])
            else:
                x2 = x1 = 0
            del_w = self.param_dict["weight_change"] * x1 * x2
            self.weight_history[step_number] = prev_w + del_w
        else:
            self.weight_history[step_number] = prev_w


class STDPSynapse:
    def __init__(self, f, t, final_timestep):
        self.pre = f
        self.post = t
        self.pre.post.append(self)
        self.post.pre.append(self)
        self.param_dict = {
            "E": -70.,
            "tau1": 1.,
            "tau2": 2.,
            "initial_weight": 1.,
            "weight_change": 0.0,
            "gmax": 1.,
            "relevant_period": 0,
            "decay": 0.05,
            "relevant_spike_no": 20,
            "sniff_cycle_period": 100,
            "STDP_slope": 20,
            "STDP_amplitude": 0.0001
        }
        self.weight_history = np.zeros(final_timestep)
        self.g_max = 1
        self.pre_firing_t = np.NINF
        self.post_firing_t = np.NINF

    def __repr__(self):
        return "" + str(self.pre.id) + " " + str(self.post.id)

    def update_starting_parameters(self, not_loaded):
        if not_loaded:
            self.weight_history[0] = self.param_dict["initial_weight"]
        t1 = self.param_dict["tau1"]
        t2 = self.param_dict["tau2"]
        if t1 == t2:
            self.g_max = 0
        elif t1 == 0:
            self.g_max = -1
        elif t2 == 0:
            self.g_max = 1
        else:
            mx = t1 * t2 * (np.log(t2) - np.log(t1)) / (t2 - t1)
            self.g_max = 1 / (np.exp(-1 * mx / t1) - np.exp(-1 * mx / t2))

    def get_input(self,  step_number, timestep):
        """
        Calculate and return the output of this synapse
        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        :return: The effect this synapse will have on its post neuron
        """
        rs = np.arange(self.pre.output_history.shape[0])[max(0, step_number - 100): step_number]\
                    [np.greater(self.pre.output_history, 0)[max(0, step_number - 100): step_number]]
        if rs.shape[0] > self.param_dict["relevant_spike_no"]:
            rs = rs[-1 * int(self.param_dict["relevant_spike_no"]):]
        rsi = -1 * timestep * (step_number - rs)

        g = self.g_max * (np.exp(rsi / self.param_dict["tau1"]) - np.exp(rsi / self.param_dict["tau2"]))
        #g = np.abs(1 * (np.exp(rsi / self.param_dict["tau1"]) - np.exp(rsi / self.param_dict["tau2"])))

        prev_v = self.post.voltage_history[step_number - 1] if self.post.output_history[step_number - 1] == 0 else \
            self.post.param_dict["resting_voltage"]

        cur = self.weight_history[step_number - 1] * np.sum(g) * (self.param_dict["E"] - prev_v)
        # if(self.post.id == 350):
        #     print(g, cur)

        return cur

    def update_weights(self,  step_number, timestep):
        """
        Use a learning rule to update the weight of this synapse
        must set self.weight_history[step_number]

        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        """
        # prev_w = ((self.weight_history[step_number - 1] - self.param_dict["initial_weight"]) *
        #          np.exp(-1 * timestep / self.param_dict["decay"])) + self.param_dict["initial_weight"]
        prev_w = self.weight_history[step_number - 1]
        self.weight_history[step_number] = prev_w + self.STDP_value(step_number)

    def STDP_value(self, step_number):
        if self.pre.output_history[step_number] or self.post.output_history[step_number]:
            if self.pre.output_history[step_number]:
                self.pre_firing_t = step_number
            if self.post.output_history[step_number]:
                self.post_firing_t = step_number

            if self.post_firing_t != np.NINF and self.pre_firing_t != np.NINF:
                t = self.pre_firing_t - self.post_firing_t
                self.post_firing_t = np.NINF
                self.pre_firing_t = np.NINF
                if t < 0:
                    return self.strengthen_STDP(t)
                elif t > 0:
                    return self.weaken_STDP(t)
        return 0

    def strengthen_STDP(self, t):
        return self.param_dict["STDP_amplitude"]*np.exp(t / self.param_dict["STDP_slope"])

    def weaken_STDP(self, t):
        return -self.param_dict["STDP_amplitude"]/2 *np.exp(-t / (2*self.param_dict["STDP_slope"]))


class SombreroSynapse:
    def __init__(self, f, t, final_timestep):
        self.pre = f
        self.post = t
        self.pre.post.append(self)
        self.post.pre.append(self)
        self.param_dict = {
            "E": -70.,
            "tau1": 1.,
            "tau2": 2.,
            "initial_weight": 1.,
            "weight_change": 0.0,
            "gmax": 1.,
            "relevant_period": 0,
            "decay": 0.05,
            "relevant_spike_no": 20,
            "sniff_cycle_period": 100,
            "exite_width": 1.5,
            "inhib_width": 4.0,
            "exite_magnitude": 1.0,
            "inhib_magnitude": 1.0
        }
        self.weight_history = np.zeros(final_timestep)
        self.g_max = 1
        self.pre_firing_t = np.NINF
        self.post_firing_t = np.NINF

    def __repr__(self):
        return "" + str(self.pre.id) + " " + str(self.post.id)

    def update_starting_parameters(self, not_loaded):
        if not_loaded:
            self.weight_history[0] = self.param_dict["initial_weight"]
        t1 = self.param_dict["tau1"]
        t2 = self.param_dict["tau2"]
        if t1 == t2:
            self.g_max = 0
        elif t1 == 0:
            self.g_max = -1
        elif t2 == 0:
            self.g_max = 1
        else:
            mx = t1 * t2 * (np.log(t2) - np.log(t1)) / (t2 - t1)
            self.g_max = 1 / (np.exp(-1 * mx / t1) - np.exp(-1 * mx / t2))

    def get_input(self,  step_number, timestep):
        """
        Calculate and return the output of this synapse
        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        :return: The effect this synapse will have on its post neuron
        """
        rs = np.arange(self.pre.output_history.shape[0])[max(0, step_number - 100): step_number]\
                    [np.greater(self.pre.output_history, 0)[max(0, step_number - 100): step_number]]
        if rs.shape[0] > self.param_dict["relevant_spike_no"]:
            rs = rs[-1 * int(self.param_dict["relevant_spike_no"]):]
        rsi = -1 * timestep * (step_number - rs)

        g = self.g_max * (np.exp(rsi / self.param_dict["tau1"]) - np.exp(rsi / self.param_dict["tau2"]))
        # g = np.abs(1 * (np.exp(rsi / self.param_dict["tau1"]) - np.exp(rsi / self.param_dict["tau2"])))

        prev_v = self.post.voltage_history[step_number - 1] if self.post.output_history[step_number - 1] == 0 else \
            self.post.param_dict["resting_voltage"]

        cur = self.weight_history[step_number - 1] * np.sum(g) * (self.param_dict["E"] - prev_v)
        # if(self.post.id == 350):
        #     print(g, cur)

        return cur

    def Sombrero_value(self, step_number):
        if self.pre.output_history[step_number] or self.post.output_history[step_number]:
            if self.pre.output_history[step_number]:
                self.pre_firing_t = step_number
            if self.post.output_history[step_number]:
                self.post_firing_t = step_number

            if self.post_firing_t != np.NINF and self.pre_firing_t != np.NINF:
                t = self.pre_firing_t - self.post_firing_t
                self.post_firing_t = np.NINF
                self.pre_firing_t = np.NINF
                ex = (self.param_dict["exite_magnitude"] + self.param_dict["inhib_magnitude"]) *\
                    np.exp(-1 * t * t / (2 * self.param_dict["exite_width"] * self.param_dict["exite_width"]))
                inhib = self.param_dict["inhib_magnitude"] *\
                    np.exp(-1 * t * t / (2 * self.param_dict["inhib_width"] * self.param_dict["inhib_width"]))
                change = ex - inhib
                # if change != 0:
                #     print("Som", t, change)
                return self.param_dict["weight_change"] * change
        return 0

    def update_weights(self,  step_number, timestep):
        # TODO: include a spike time dependant learning rule (spike timing dependant plasticity)
        """
        Use a learning rule to update the weight of this synapse
        must set self.weight_history[step_number]

        :param step_number: the timestep number of this call of the function
        :param timestep: the size of one timestep in ms
        """
        #prev_w = ((self.weight_history[step_number - 1] - self.param_dict["initial_weight"]) *
        #          np.exp(-1 * timestep / self.param_dict["decay"])) + self.param_dict["initial_weight"]
        prev_w = self.weight_history[step_number - 1]
        self.weight_history[step_number] = prev_w + self.Sombrero_value(step_number)


synapse_names = {
    "basicS": Synapse,
    "no_fireS": NonfireSynapse,
    "STDPS": STDPSynapse,
    "hat": SombreroSynapse
}
