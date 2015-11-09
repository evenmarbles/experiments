import sys
import argparse
import random

import numpy as np
from matplotlib import pyplot as plt

from mlpy.auxiliary.io import load_from_file
from mlpy.mdp.stateaction import MDPState
from mlpy.mdp.continuous import CbTData, CASML


def plot_sampled(obs, sampled):
    fig = plt.figure()
    plt.rcParams['legend.fontsize'] = 10
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    markers = ['o', 'v', '<', '>', '8', 's', 'p', '*', 'x', 'D']
    colors = ['b', 'g', 'r', 'c', 'm', 'k']

    for i, o in enumerate(obs):
        c = random.choice(colors)
        m = random.choice(markers)
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
    fig.show()


def evaluate_action(actions, obs, sampled, plot=False):
    n_samples = obs.shape[0]
    d, n = obs[0].shape

    fig = None
    if plot:
        fig = plt.figure()
        plt.rcParams['legend.fontsize'] = 10

    obs_avg = np.zeros((n, d))
    for i, o in enumerate(obs):
        obs_avg += o.T

        if plot:
            ax = fig.add_subplot((n_samples + 1) / 2, 2, i + 1, projection='3d')

            a_pt = np.zeros((n, d))
            o_pt = np.zeros((n, d))
            s_pt = np.zeros((n, d))
            a_pt[0] = o[:, 0]
            o_pt[0] = o[:, 0]
            s_pt[0] = o[:, 0]
            ax.scatter(o[:, 0][0], o[:, 0][1], o[:, 0][2], edgecolors='g', c='g', marker='o')

            for j, a in enumerate(actions.T):
                a_pt[j + 1] = a_pt[j] + a
                ax.scatter(a_pt[j + 1][0], a_pt[j + 1][1], a_pt[j + 1][2], edgecolors='y', c='y', marker='o')

                o_pt[j + 1] = o_pt[j] + o[:, j + 1] - o[:, j]
                ax.scatter(o_pt[j + 1][0], o_pt[j + 1][1], o_pt[j + 1][2], edgecolors='k', c='k', marker='o')

                s_pt[j + 1] = s_pt[j] + sampled[:, j + 1] - sampled[:, j]
                ax.scatter(s_pt[j + 1][0], s_pt[j + 1][1], s_pt[j + 1][2], edgecolors='r', c='r', marker='o')

            ax.plot(a_pt.T[0], a_pt.T[1], a_pt.T[2], c='y', label='action')
            ax.plot(o_pt.T[0], o_pt.T[1], o_pt.T[2], c='k', label='observed')
            ax.plot(s_pt.T[0], s_pt.T[1], s_pt.T[2], c='r', label='sampled')
            ax.legend()

            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.set_zlabel('Z position')
            ax.set_title('Comparison of trajectories:\n True action - observed (#{0}) - sampled'.format(i + 1))

    if plot:
        fig.show()

    obs_avg /= float(n_samples)

    obs_error = np.zeros(n - 1)
    sampled_error = np.zeros(n - 1)

    for i, a in enumerate(actions.T):
        obs_delta = obs_avg[i + 1] - obs_avg[i]
        obs_error[i] = np.linalg.norm(obs_delta - a)

        sampled_delta = sampled[:, i + 1] - sampled[:, i]
        sampled_error[i] = np.linalg.norm(sampled_delta - a)

    avg_obs_error = np.sum(obs_error) / float(n - 1)
    avg_sampled_error = np.sum(sampled_error) / float(n - 1)

    error = abs(avg_obs_error - avg_sampled_error)

    return error, avg_obs_error, avg_sampled_error


def evaluate_delta(obs, sampled, plot=False):
    fig = None
    ax1 = None
    ax2 = None

    markers = ['o', 'v', '<', '>', '8', 's', 'p', '*', 'x', 'D']
    colors = ['b', 'g', 'r', 'c', 'm', 'k']

    if plot:
        fig = plt.figure()
        plt.rcParams['legend.fontsize'] = 10
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    n_samples = obs.shape[0]
    d, n = obs[0].shape

    obs_avg = np.zeros((n, d))
    for i, o in enumerate(obs):
        obs_avg += o.T

        if plot:
            c = random.choice(colors)
            m = random.choice(markers)
            for t in o.T:
                [x, y, z] = t
                ax2.scatter(x, y, z, edgecolors=c, c=c, marker=m)

            ax2.plot(o[0], o[1], o[2], c=c, label='trajectory {0}'.format(i + 1))

    obs_avg /= float(n_samples)

    error = np.zeros(n - 1)
    for i in xrange(n - 1):
        obs_delta = obs_avg[i + 1] - obs_avg[i]
        sampled_delta = sampled[:, i + 1] - sampled[:, i]
        error[i] = np.linalg.norm(obs_delta - sampled_delta)

        if plot:
            [x, y, z] = obs_avg[i]
            ax1.scatter(x, y, z, edgecolors='g', c='g', marker='v')

            [sx, sy, sz] = sampled[:, i]
            ax1.scatter(sx, sy, sz, edgecolors='y', c='y', marker='^')
            ax2.scatter(sx, sy, sz, edgecolors='y', c='y', marker='^')

    if plot:
        ax1.plot(obs_avg.T[0], obs_avg.T[1], obs_avg.T[2], c='g', label='observed (avg)')
        ax1.plot(sampled[0], sampled[1], sampled[2], c='y', label='sampled')
        ax1.legend()

        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.set_zlabel('Z position')
        ax1.set_title(
            'Comparison of observed trajectories averaged over # of trajectories and\n \
the sampled trajectory (# trajectories: {0})'.format(n_samples))

        ax2.plot(sampled[0], sampled[1], sampled[2], c='y', label='sampled')
        ax2.legend()

        ax2.set_xlabel('X position')
        ax2.set_ylabel('Y position')
        ax2.set_zlabel('Z position')
        ax2.set_title(
            'Comparison of observed trajectories and the sampled trajectory')

        fig.show()

    avg_error = np.sum(error) / float(n - 1)

    return avg_error


def main(args):
    try:
        data = load_from_file(args.infile)
        obs = data["state"]
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
    model = CASML(CbTData(case_t_template, rho=args.rho, tau=args.tau, sigma=args.sigma),
                  ncomponents=args.ncomponents)

    n = obs.shape[0]
    action_error = -np.inf * np.ones(n)
    delta_error = -np.inf * np.ones(n)

    for i, states in enumerate(obs):
        # Train CASML's case base and hmm with states and actions
        model.fit(states, actions)

        # Test model
        cntr = 0
        iter_ = 0
        while cntr < 10:
            sampled = None
            try:
                sampled = np.array([model.sample()]).T

                for iter_, a in enumerate(actions.T):
                    # sample next state resulting from executing action `a` in state `state`
                    next_state = model.sample(MDPState(sampled[:, -1]), a)[:, np.newaxis]
                    sampled = np.hstack([sampled, next_state])
            except:
                print "{0}:{1} Failed to infer next state distribution at step {2}.".format(i + 1, cntr + 1, iter_ + 1)
                # plot_sampled(obs[0:i+1], sampled)
                cntr += 1
                continue
            break

        if cntr < 10:
            action_error[i] = evaluate_action(actions, obs[0:i + 1], sampled, plot=True)[0]
            delta_error[i] = evaluate_delta(obs[0:i + 1], sampled, plot=True)

    print "Error to true action:\n{0}".format({k: e for k, e in enumerate(action_error)})
    print "Error to average trajectory:\n{0}".format({k: e for k, e in enumerate(delta_error)})
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment: Learn continuous action and state model for motion.")
    ap.add_argument("--rho", type=float, default=0.97, required=False, help="The maximum error rho.")
    ap.add_argument("--tau", type=float, default=0.005, required=False, help="The maximum error tau.")
    ap.add_argument("--sigma", type=float, default=0.001, required=False, help="The maximum error sigma.")
    ap.add_argument("--ncomponents", type=int, default=2, required=False, help="The number of hidden states.")
    ap.add_argument("--retrieval_method", type=str, default='radius-n', required=False,
                    help="The state retrieval method.")
    ap.add_argument("--retrieval_method_params", type=float, default=0.01, required=False,
                    help="The retrieval method parameters.")
    ap.add_argument("--infile", type=str, required=True, help="The trajectory data file name.")
    ap.add_argument("--policy", type=str, required=True, help="The policy file name.")
    ap.add_argument("--policy_num", type=str, default=0, required=False,
                    help="The identification of the policy to run")

    main(ap.parse_args())
