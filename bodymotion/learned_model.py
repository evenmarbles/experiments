import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mlpy.auxiliary.io import load_from_file, save_to_file


def calc_stats(obs):
    nobs = obs.shape[0]
    d, n = obs[0].shape

    obs_avg = np.zeros((n, d))
    minmax = np.zeros((n, d), dtype=np.object)
    hist = np.zeros((n, d), dtype=np.object)
    edges = np.zeros((n, d), dtype=np.object)

    for i, o in enumerate(obs):
        obs_avg += o.T

    obs_avg /= float(nobs)

    o = np.array(list(obs), dtype=np.float).T

    for i, t in enumerate(o):
        hist[i, 0], edges[i, 0] = np.histogram(t[0])
        hist[i, 1], edges[i, 1] = np.histogram(t[1])
        hist[i, 2], edges[i, 2] = np.histogram(t[2])

        minmax[i, 0] = (t[0].min(), t[0].max())
        minmax[i, 1] = (t[1].min(), t[1].max())
        minmax[i, 2] = (t[2].min(), t[2].max())

    return obs_avg, minmax, hist, edges


def calc_histogram(obs):
    o = np.array(list(obs), dtype=np.float).T
    x_hist, x_edges = np.histogram(o[:, 0], bins=10)
    y_hist, y_edges = np.histogram(o[:, 1], bins=10)
    z_hist, z_edges = np.histogram(o[:, 2], bins=10)

    return x_hist, y_hist, z_hist, x_edges, y_edges, z_edges


def plot_pos_error(obs, sampled, fig=None, ax=None):
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure()
        plt.rcParams['legend.fontsize'] = 10
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        fig.show()

    n, d = obs.shape

    ax.cla()
    ax.plot(obs.T[0], obs.T[1], obs.T[2], c='g', label='observed (avg)')
    ax.plot(sampled[0], sampled[1], sampled[2], c='y', label='sampled')
    for i in xrange(n - 1):
        [x, y, z] = obs[i]
        ax.scatter(x, y, z, edgecolors='g', c='g', marker='v')

        [sx, sy, sz] = sampled[:, i]
        ax.scatter(sx, sy, sz, edgecolors='y', c='y', marker='^')

    ax.legend()

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')
    ax.set_title('Comparison of averaged observed trajectories and the sampled trajectory')

    fig.canvas.draw()

    return fig, ax


def evaluate_pos_error(obs, sampled):
    n = obs.shape[0]

    error = np.zeros(n)
    for i in xrange(n):
        error[i] = np.linalg.norm(obs[i] - sampled[:, i])

    avg_error = np.sum(error) / float(n)
    return avg_error, error


def evaluate_action(actions, obs, sampled):
    n = obs.shape[0]

    obs_error = np.zeros(n - 1)
    sampled_error = np.zeros(n - 1)

    for i, a in enumerate(actions.T):
        obs_delta = obs[i + 1] - obs[i]
        obs_error[i] = np.linalg.norm(obs_delta - a)

        sampled_delta = sampled[:, i + 1] - sampled[:, i]
        sampled_error[i] = np.linalg.norm(sampled_delta - a)

    avg_obs_error = np.sum(obs_error) / float(n - 1)
    avg_sampled_error = np.sum(sampled_error) / float(n - 1)

    error = abs(avg_obs_error - avg_sampled_error)

    return error, avg_obs_error, avg_sampled_error


def evaluate_delta(obs, sampled):
    n = obs.shape[0]

    error = np.zeros(n - 1)
    for i in xrange(n - 1):
        obs_delta = obs[i + 1] - obs[i]
        sampled_delta = sampled[:, i + 1] - sampled[:, i]
        error[i] = np.linalg.norm(obs_delta - sampled_delta)

    avg_error = np.sum(error) / float(n - 1)

    return avg_error


def visualize_obs(obs):
    fig = plt.figure()
    plt.rcParams['legend.fontsize'] = 10
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    markers = ['o', 'v', '<', '>', '8', 's', 'p', '*', 'x', 'D']
    colors = ['b', 'g', 'r', 'c', 'm', 'k']

    nobs = obs.shape[0]
    d, n = obs[0].shape

    obs_avg = np.zeros((n, d))
    for i, o in enumerate(obs):
        obs_avg += o.T

    # c = random.choice(colors)
    #     m = random.choice(markers)
    #     for t in o.T:
    #         [x, y, z] = t
    #         ax1.scatter(x, y, z, edgecolors=c, c=c, marker=m)
    #
    #     ax1.plot(o[0], o[1], o[2], c=c, label='trajectory {0}'.format(i + 1))
    #
    # ax1.legend()
    #
    # ax1.set_xlabel('X position')
    # ax1.set_ylabel('Y position')
    # ax1.set_zlabel('Z position')
    # ax1.set_title(
    #     'Comparison of observed trajectories and the sampled trajectory')

    obs_avg /= float(nobs)

    ax2.plot(obs_avg.T[0], obs_avg.T[1], obs_avg.T[2], c='g', label='observed (avg)')
    ax2.legend()

    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    ax2.set_zlabel('Z position')
    ax2.set_title(
        'Comparison of observed trajectories averaged over # of trajectories and\n \
the sampled trajectory (# trajectories: {0})'.format(nobs))

    fig.show()


def plot_sampled_3d(obs, sampled, fig=None, ax=None):
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure()
        plt.rcParams['legend.fontsize'] = 10
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        fig.show()

    ax.cla()

    for i, o in enumerate(obs):
        for t in o.T:
            [x, y, z] = t
            ax.scatter(x, y, z, edgecolors='b', c='b', marker='o')

        ax.plot(o[0], o[1], o[2], c='b', label='trajectory {0}'.format(i + 1))

    for s in sampled.T:
        ax.scatter(s[0], s[1], s[2], edgecolors='y', c='y', marker='^')

    ax.plot(sampled[0], sampled[1], sampled[2], c='y', label='sampled')
    ax.legend()

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')
    ax.set_title('Failed: Observed and sampled trajectories')

    fig.canvas.draw()

    return fig, ax


def plot_sampled_2d(obs, sampled, fig=None, ax1=None, ax2=None, ax3=None):
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure()
        plt.rcParams['legend.fontsize'] = 10
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        fig.show()

    ax1.cla()
    ax2.cla()
    ax3.cla()

    for i, o in enumerate(obs):
        for t in o.T:
            [x, y, z] = t
            ax1.scatter(x, y, edgecolors='b', c='b', marker='o')
            ax2.scatter(x, z, edgecolors='b', c='b', marker='o')
            ax3.scatter(y, z, edgecolors='b', c='b', marker='o')

        ax1.plot(o[0], o[1], c='b', label='trajectory {0}'.format(i + 1))
        ax2.plot(o[0], o[2], c='b', label='trajectory {0}'.format(i + 1))
        ax3.plot(o[1], o[2], c='b', label='trajectory {0}'.format(i + 1))

    for s in sampled.T:
        ax1.scatter(s[0], s[1], edgecolors='y', c='y', marker='^')
        ax2.scatter(s[0], s[2], edgecolors='y', c='y', marker='^')
        ax3.scatter(s[1], s[2], edgecolors='y', c='y', marker='^')

    ax1.plot(sampled[0], sampled[1], c='y', label='sampled')
    ax2.plot(sampled[0], sampled[2], c='y', label='sampled')
    ax3.plot(sampled[1], sampled[2], c='y', label='sampled')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('Observed and sampled trajectories: X, Y')

    ax2.set_xlabel('X position')
    ax2.set_ylabel('Z position')
    ax2.set_title('Observed and sampled trajectories: X, Z')

    ax3.set_xlabel('Y position')
    ax3.set_ylabel('Z position')
    ax3.set_title('Observed and sampled trajectories: Y, Z')

    fig.canvas.draw()

    return fig, ax1, ax2, ax3


def main(args):
    try:
        data = load_from_file(args.policy)
        actions = data['act'][args.policy_num]
    except IOError:
        sys.exit(sys.exc_info()[1])
    except KeyError, e:
        sys.exit("Key not found: {0}".format(e))

    try:
        path, filename = os.path.split(args.infile)

        filename = path + '/validation.pkl'
        data = load_from_file(filename)
        obs_avg = data["obs_avg"]
        minmax = data["minmax"]
        hist = data["hist"]
        edges = data["edges"]

        ntrials = 50
        nobs = 0
        n = 0

        minmax_error = np.zeros(ntrials)
        pos_error = np.zeros(ntrials)
        action_error = np.zeros(ntrials)
        delta_error = np.zeros(ntrials)

        sampled = None
        sampled_hist = None

        fig1 = None
        fig2 = None
        ax1 = None
        ax2 = None
        ax3 = None
        ax4 = None

        filename, extension = os.path.splitext(args.infile)
        for i in xrange(ntrials):
            data = load_from_file(filename + str(i) + extension)

            if sampled is None:
                nobs = 50   # data["sampled"].shape[0]
                d, n = data["sampled"][0].shape
                sampled_hist = np.zeros((n, d), dtype=np.object)

            sampled = np.sum(data["sampled"][:50], axis=0)
            sampled /= nobs
            fig1, ax1 = plot_sampled_3d([obs_avg.T], sampled, fig1, ax1)
            fig2, ax2, ax3, ax4 = plot_sampled_2d([obs_avg.T], sampled, fig2, ax2, ax3, ax4)

            pos_error[i] += evaluate_pos_error(obs_avg, sampled)[0]
            action_error[i] += evaluate_action(actions, obs_avg, sampled)[0]
            delta_error[i] += evaluate_delta(obs_avg, sampled)

            minmax_ = 0
            for j in xrange(n):
                [x, y, z] = sampled[:, j]
                err = np.zeros(3)
                if not minmax[j][0][0] < x < minmax[j][0][1]:
                    err[0] = abs(minmax[j][0][0] - x) if x < minmax[j][0][0] else abs(minmax[j][0][1] - x)
                if not minmax[j][1][0] < y < minmax[j][1][1]:
                    err[1] = abs(minmax[j][1][0] - y) if y < minmax[j][1][0] else abs(minmax[j][1][1] - y)
                if not minmax[j][2][0] < z < minmax[j][2][1]:
                    err[2] = abs(minmax[j][2][0] - z) if z < minmax[j][2][0] else abs(minmax[j][2][1] - z)
                err = np.sqrt(np.sum(np.square(err)))
                minmax_ += err
            minmax_error[i] += minmax_ / float(nobs)

            s = np.array(list(data["sampled"]), dtype=np.float).T

            for j, t in enumerate(s):
                sampled_hist[j, 0] = np.histogram(t[0], edges[j, 0])[0]
                sampled_hist[j, 1] = np.histogram(t[1], edges[j, 1])[0]
                sampled_hist[j, 2] = np.histogram(t[2], edges[j, 2])[0]

    except IOError:
        sys.exit(sys.exc_info()[1])
    except KeyError, e:
        sys.exit("Key not found: {0}".format(e))

    pos_error /= float(nobs)
    action_error /= float(nobs)
    delta_error /= float(nobs)
    minmax_error /= float(nobs)

    save_to_file(args.outfile, {
        "pos_error": pos_error,
        "action_error": action_error,
        "delta_error": delta_error,
    })

    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment: Learn continuous action and state model for motion.")
    ap.add_argument("--collect_data", action="store_true",
                    help="When set, data is being collected and saved to file.")
    ap.add_argument("--rho", type=float, default=0.97, required=False, help="The maximum error rho.")
    ap.add_argument("--tau", type=float, default=0.005, required=False, help="The maximum error tau.")
    ap.add_argument("--sigma", type=float, default=0.001, required=False, help="The maximum error sigma.")
    ap.add_argument("--ncomponents", type=int, default=10, required=False, help="The number of hidden states.")
    ap.add_argument("--retrieval_method", type=str, default='radius-n', required=False,
                    help="The state retrieval method.")
    ap.add_argument("--retrieval_method_params", type=float, default=0.025, required=False,
                    help="The retrieval method parameters.")
    ap.add_argument("--infile", type=str, required=True, help="The trajectory data file name.")
    ap.add_argument("--policy", type=str, required=True, help="The policy file name.")
    ap.add_argument("--policy_num", type=str, default=0, required=False,
                    help="The identification of the policy to run")
    ap.add_argument("--outfile", type=str, required=False, help="The collected data file name.")

    args_ = ap.parse_args()

    if args_.collect_data:
        if args_.outfile is None:
            ap.error("with --collect_data, --outfile is required")

    main(args_)
