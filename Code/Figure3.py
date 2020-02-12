from header import *


def get_results(x_param, A, t, end_time, Omega, kappa, delta, b):
    start_index = int(len(t) / 5)
    end_index = int(len(t) * 4 / 5)
    num_channels = len(A)
    delta_t = t[1] - t[0]
    tem_mult = timeEncoder(kappa, delta, b, A)
    spikes = tem_mult.encode_precise(x_param, Omega, end_time, tol=1e-10)
    rec_mult = tem_mult.decode_mixed(
        spikes, t, x_param[0].get_sinc_locs(), Omega, delta_t
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
        x_param.append(bandlimitedSignal(Omega))
        x_param[-1].random(t, padding=sinc_padding)
    return x_param


def get_params_for_spike_rate(x_param, t, A, end_time, num_spikes):
    n_signals = len(x_param)
    x_ints = np.zeros((n_signals, 1))
    for n in range(n_signals):
        x_ints[n] = x_param[n].get_precise_integral(0, end_time)
    y_ints = np.array(A).dot(x_ints)
    n_channels = len(A)
    b = [1, 1, 1]
    kappa = [1] * n_channels
    delta = [1] * n_channels
    kappadelta = [2 * kappa[l] * delta[l] for l in range(n_channels)]
    for m in range(n_channels):
        needed_integral = (num_spikes[m] + 0.6) * kappadelta[m]
        b[m] = (needed_integral - y_ints[m, 0]) / end_time
    return b, kappa, delta


def GetData():
    # Settings for x_param
    end_time = 20
    sinc_padding = 2
    delta_t = 1e-4
    t = np.arange(0, end_time + delta_t, delta_t)
    Omega = np.pi
    seed = 0
    np.random.seed(int(seed))
    num_signals = 2
    num_sincs = end_time - sinc_padding * 2
    total_deg_freedom = num_sincs * num_signals

    # Settings for time encoding machine
    num_channels = 3
    A = [[0.4, 1], [0.3, -0.2], [1, 0.7]]

    num_constraints_1 = 12
    num_constraints_2 = 8
    num_constraints_3_range = np.arange(2, 21, 1)
    n_constraints_total = np.zeros_like(num_constraints_3_range)
    n_constraints_const = np.zeros_like(num_constraints_3_range)
    for n_s_r in range(len(num_constraints_3_range)):
        n_constraints_const[n_s_r] = (
            min(num_constraints_1, num_sincs)
            + min(num_constraints_2, num_sincs)
            + min(num_constraints_3_range[n_s_r], num_sincs)
        )
        n_constraints_total[n_s_r] = (
            num_constraints_1 + num_constraints_2 + num_constraints_3_range[n_s_r]
        )

    # Settings for Simulation
    num_trials = 100
    results = np.zeros((num_trials, num_signals, len(num_constraints_3_range)))

    for n_t in range(num_trials):
        x_param = create_signal(num_signals, t, delta_t, Omega, sinc_padding)
        for n_s_r in range(len(num_constraints_3_range)):
            b, kappa, delta = get_params_for_spike_rate(
                x_param,
                t,
                A,
                end_time,
                [num_constraints_1, num_constraints_2, num_constraints_3_range[n_s_r]],
            )
            res1, res2 = get_results(x_param, A, t, end_time, Omega, kappa, delta, b)
            results[n_t, 0, n_s_r] = res1
            results[n_t, 1, n_s_r] = res2

    data_filename = Data_Path + "Figure3.pkl"
    with open(data_filename, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(
            [n_constraints_total, n_constraints_const, results, total_deg_freedom], f
        )


def GenerateFigure():

    data_filename = Data_Path + "Figure3.pkl"
    figure_filename = Figure_Path + "Figure3.png"

    with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
        obj = pickle.load(f, encoding="latin1")

    n_constraints_total = obj[0]
    n_constraints_const = obj[1]
    results = obj[2]
    num_signals = results.shape[1]
    total_deg_freedom = obj[3]

    plt.rc("text", usetex=True)

    clr = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(8, 3))
    plt.plot(n_constraints_total, np.mean(results[:, 0, :], 0), label=r"$x^{(0)}(t)$")
    plt.plot(n_constraints_total, np.mean(results[:, 1, :], 0), label=r"$x^{(1)}(t)$")
    plt.xlabel("Total number of constraints")
    plt.ylabel("Reconstruction error")
    plt.legend(loc="best")
    ax = plt.gca()
    ax.set_yscale("log")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax.set_xticks(n_constraints_total[0::2])
    ax.set_xticklabels(n_constraints_total[0::2])
    ax2.set_xticks(n_constraints_total[0::2])
    ax2.set_xticklabels(n_constraints_const[0::2])
    ax = plt.gca()
    ax.set_yscale("log")
    ax2.axvline(total_deg_freedom, color="red")
    plt.xlabel("Number of useful constraints")
    plt.ylabel("Reconstruction Error")
    plt.tight_layout()
    plt.savefig(figure_filename)


if __name__ == "__main__":
    data_filename = Data_Path + "Figure3.pkl"

    if not os.path.isfile(data_filename):
        GetData()
    if graphical_import:
        GenerateFigure()
