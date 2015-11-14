import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from mlpy.auxiliary.io import load_from_file, save_to_file
from mlpy.mdp.stateaction import MDPState
from mlpy.mdp.continuous import CbTData, CASML
from mlpy.tools.misc import Timer


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


def plot_sampled(obs, sampled, fig=None, ax=None):
    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure()
        plt.rcParams['legend.fontsize'] = 10
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        fig.show()

    ax.cla()
    markers = ['o', 'v', '<', '>', '8', 's', 'p', '*', 'x', 'D']
    colors = ['b', 'g', 'r', 'c', 'm', 'k']

    for i, o in enumerate(obs):
        c = np.random.choice(colors)
        m = np.random.choice(markers)
        for t in o.T:
            [x, y, z] = t
            ax.scatter(x, y, z, edgecolors=c, c=c, marker=m)

        ax.plot(o[0], o[1], o[2], c=c, label='trajectory {0}'.format(i + 1))

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


def main(args):
    if args.collect_data:
        try:
            data = load_from_file(args.infile)
            train = data["train"]
            test = data["test"]
        except IOError:
            sys.exit(sys.exc_info()[1])
        except KeyError, e:
            sys.exit("Key not found: {0}".format(e))

        try:
            data = load_from_file(args.policy)
            actions = data['act'][args.policy_num]
        except IOError:
            sys.exit(sys.exc_info()[1])
        except KeyError, e:
            sys.exit("Key not found: {0}".format(e))

        case_t_template = {
            "state": {
                "type": "float",
                "value": "data.state",
                "is_index": True,
                "retrieval_method": args.retrieval_method,
                "retrieval_method_params": args.retrieval_method_params
            },
            "act": {
                "type": "float",
                "value": "data.action",
                "is_index": False,
                "retrieval_method": "cosine",
            },
            "delta_state": {
                "type": "float",
                "value": "data.next_state - data.state",
                "is_index": False,
            }
        }

        ntrials = train.shape[0]
        nobs = train[0].shape[0]
        d, n = train[0][0].shape

        sampled = np.zeros((nobs, d, n))
        failed = np.zeros(nobs, dtype=np.int)

        for i in xrange(0, ntrials):
            model = CASML(CbTData(case_t_template, rho=args.rho, tau=args.tau, sigma=args.sigma,
                                  plot_reuse=False, plot_reuse_params='original_origin'),
                          ncomponents=args.ncomponents)

            with Timer() as tm:
                for j, states in enumerate(train[i]):
                    # Train CASML's case base and hmm with states and actions
                    model.fit(states, actions)

                    # Test model
                    iter_ = 0
                    while failed[j] < 10:
                        try:
                            sampled[j, :, 0] = model.sample()

                            for iter_, a in enumerate(actions.T):
                                # sample next state resulting from executing action `a` in state `state`
                                next_state = model.sample(MDPState(sampled[j, :, iter_]), a)
                                if next_state is None:
                                    raise TypeError
                                sampled[j, :, iter_ + 1] = next_state
                        except:
                            # plot_sampled(obs[:j], sampled[i, j, :, :iter_])

                            sampled[j, :].fill(0)
                            failed[j] += 1

                            print "{0}:{1} Failed to infer next state distribution at step {2}.".format(j,
                                                                                                        failed[j],
                                                                                                        iter_)
                            continue
                        break

            print('Request took %.03f sec.' % tm.time)

            filename, extension = os.path.splitext(args.outfile)
            save_to_file(filename + str(i) + extension, {
                "model": model,
                "sampled": sampled,
                "failed": failed,
                "time": tm.time,
            })

            sampled.fill(0)
            failed.fill(0)

        obs_avg, minmax, hist, edges = calc_stats(test)
        # x_hist, y_hist, z_hist, x_edges, y_edges, z_edges = calc_histogram(obs[:50])

        path, filename = os.path.split(args.outfile)
        filename = path + '/validation.pkl'
        save_to_file(filename, {
            "obs_avg": obs_avg,
            "minmax": minmax,
            "hist": hist,
            "edges": edges,
        })

        return

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

        nobs = 50
        pos_error = np.zeros(nobs)
        action_error = np.zeros(nobs)
        delta_error = np.zeros(nobs)

        failed = np.zeros(nobs, dtype=np.int)
        minmax_error = np.zeros(nobs)
        # minmax_error = None
        sampled = None
        hist = None
        ncases = 0

        fig = None
        ax = None

        n = 0

        filename, extension = os.path.splitext(args.infile)
        for i in xrange(50):
            data = load_from_file(filename + str(i) + extension)
            failed += data["failed"]
            ncases += (data["model"]._cb_t._counter - 1)

            if sampled is None:
                d, n = data["sampled"][0].shape
                sampled = np.zeros((d, n))
                hist = np.zeros((n, d), dtype=np.object)

            for j in xrange(nobs):
                pos_error[j] += evaluate_pos_error(obs_avg, data["sampled"][j])[0]
                action_error[j] += evaluate_action(actions, obs_avg, data["sampled"][j])[0]
                delta_error[j] += evaluate_delta(obs_avg, data["sampled"][j])

                minmax_ = 0
                for k in xrange(n):
                    [x, y, z] = data["sampled"][j, :, k]
                    err = np.zeros(3)
                    if not minmax[k][0][0] < x < minmax[k][0][1]:
                        err[0] = abs(minmax[k][0][0] - x) if x < minmax[k][0][0] else abs(minmax[k][0][1] - x)
                    if not minmax[k][1][0] < y < minmax[k][1][1]:
                        err[1] = abs(minmax[k][1][0] - y) if y < minmax[k][1][0] else abs(minmax[k][1][1] - y)
                    if not minmax[k][2][0] < z < minmax[k][2][1]:
                        err[2] = abs(minmax[k][2][0] - z) if z < minmax[k][2][0] else abs(minmax[k][2][1] - z)
                    err = np.sqrt(np.sum(np.square(err)))
                    minmax_ += err
                minmax_error[j] += minmax_ / float(nobs)

                # fig, ax = plot_sampled([obs_avg.T], data["sampled"][j], fig, ax)

            # hist[i, 0], edges[i, 0] = np.histogram(t[0])
            # hist[i, 1], edges[i, 1] = np.histogram(t[1])
            # hist[i, 2], edges[i, 2] = np.histogram(t[2])

            sampled += data["sampled"][-1]

        sampled /= nobs
        # plot_sampled([obs_avg.T], sampled)

    except IOError:
        sys.exit(sys.exc_info()[1])
    except KeyError, e:
        sys.exit("Key not found: {0}".format(e))

    failed /= float(nobs)
    pos_error /= float(nobs)
    action_error /= float(nobs)
    delta_error /= float(nobs)
    minmax_error /= float(nobs)
    ncases /= nobs

    save_to_file(args.outfile, {
        "failed": failed,
        "pos_error": pos_error,
        "action_error": action_error,
        "delta_error": delta_error,
        "ncases": ncases
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
