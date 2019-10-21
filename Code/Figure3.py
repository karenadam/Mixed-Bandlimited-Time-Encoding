from header import *


def get_results(x_param, A, t, end_time, Omega, kappa, delta, b):
    start_index = int(len(t) / 5)
    end_index = int(len(t) * 4 / 5)
    num_channels = len(A)
    delta_t = t[1] - t[0]
    tem_mult = timeEncoder(kappa, delta, b, n_channels=num_channels)
    spikes = tem_mult.encode_mixed_precise(x_param, A, Omega, end_time)
    rec_mult = tem_mult.decode_multi_signal_multi_channel_recursive(
        spikes, A, t, Omega, delta_t, num_iterations=100
    )
    x_0 = x_param[0].sample(t)
    x_1 = x_param[1].sample(t)
    res1 = np.mean(((rec_mult[0, :] - x_0) ** 2)[start_index:end_index]) / np.mean(
        x_0[start_index:end_index] ** 2
    )
    res2 = np.mean(((rec_mult[1, :] - x_1) ** 2)[start_index:end_index]) / np.mean(
        x_1[start_index:end_index] ** 2
    )
    return res1, res2


def create_signal(num_signals, t, delta_t, Omega, sinc_padding):
    x = np.zeros((num_signals, len(t)))
    x_param = []
    for n in range(num_signals):
        x_param.append(bandlimitedSignal(t, delta_t, Omega, padding=sinc_padding))
        x[n, :] = x_param[-1].sample(t)
    return x, x_param


def get_params_for_spike_rate(x_param, t, A, end_time, spike_rates):
    n_signals = len(x_param)
    x_ints = np.zeros((n_signals, 1))
    for n in range(n_signals):
        x_ints[n] = x_param[n].get_total_integral(t)
    y_ints = np.array(A).dot(x_ints)
    n_channels = len(A)
    b = [1, 1, 1]
    kappa = [1] * n_channels
    delta = [1] * n_channels
    kappadelta = [2 * kappa[l] * delta[l] for l in range(n_channels)]
    for m in range(n_channels):
        needed_integral = (spike_rates[m] * end_time + 1.5) * kappadelta[m]
        b[m] = (needed_integral - y_ints[m, 0]) / end_time
    return b, kappa, delta


def GetData():
    # Settings for x
    end_time = 20
    sinc_padding = 3
    delta_t = 1e-4
    t = np.arange(0, end_time + delta_t, delta_t)
    Omega = np.pi
    seed = 0
    np.random.seed(int(seed))
    num_signals = 2
    rOmega = Omega / np.pi

    # Settings for time encoding machine
    num_channels = 3
    A = [[0.4, 1], [0.3, -0.2], [1, 0.7]]
    spike_rate1 = 0.5
    spike_rate2 = 0.75
    spike_range = np.arange(0.1, 1.7, 0.1)
    n_spikes_total = np.zeros_like(spike_range)
    n_spikes_const = np.zeros_like(spike_range)
    for n_s_r in range(len(spike_range)):
        n_spikes_const[n_s_r] = round(
            min(spike_rate1, rOmega)
            + min(spike_rate2, rOmega)
            + min(spike_range[n_s_r], rOmega),
            2,
        )
        n_spikes_total[n_s_r] = round(spike_rate1 + spike_rate2 + spike_range[n_s_r], 2)

    # Settings for Simulation
    num_trials = 100
    results = np.zeros((num_trials, num_signals, len(spike_range)))

    for n_t in range(num_trials):
        print(n_t)
        x, x_param = create_signal(num_signals, t, delta_t, Omega, sinc_padding)
        for n_s_r in range(len(spike_range)):
            b, kappa, delta = get_params_for_spike_rate(
                x_param, t, A, end_time, [spike_rate1, spike_rate2, spike_range[n_s_r]]
            )
            res1, res2 = get_results(x_param, A, t, end_time, Omega, kappa, delta, b)
            results[n_t, 0, n_s_r] = res1
            results[n_t, 1, n_s_r] = res2

    filename = "../Data/Figure3.pkl"
    with open(filename, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump([n_spikes_total, n_spikes_const, results, rOmega], f)


def GenerateFigure():

    data_filename = "../Data/Figure3.pkl"
    with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
        obj = pickle.load(f, encoding="latin1")

    n_spikes_total = obj[0]
    n_spikes_const = obj[1]
    results = obj[2]
    num_signals = results.shape[1]
    rOmega = obj[3]

    plt.rc("text", usetex=True)

    clr = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(8, 3))
    plt.plot(n_spikes_total, np.mean(results[:, 0, :], 0), label=r'$x^{(0)}(t)$')
    plt.plot(n_spikes_total, np.mean(results[:, 1, :], 0), label=r'$x^{(1)}(t)$')
    plt.xlabel('Total spiking rate')
    plt.ylabel('Reconstruction error')
    plt.legend(loc = 'best')
    ax = plt.gca()
    ax.set_yscale("log")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax.set_xticks(n_spikes_total[0::2])
    ax.set_xticklabels(n_spikes_total[0::2])
    ax2.set_xticks(n_spikes_total[0::2])
    ax2.set_xticklabels(n_spikes_const[0::2])
    ax = plt.gca()
    ax.set_yscale("log")
    plt.axvline(num_signals * rOmega, color="red")
    plt.xlabel('Constrained spiking rate')
    plt.ylabel('Reconstruction Error')
    plt.savefig("../Figures/Figure3.png")


if __name__ == "__main__":
    data_filename = "../Data/Figure3.pkl"

    if not os.path.isfile(data_filename):
        GetData()
    if graphical_import:
        GenerateFigure()
