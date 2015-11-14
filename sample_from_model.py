import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from mlpy.auxiliary.io import load_from_file, save_to_file
from mlpy.tools.misc import Timer
from mlpy.mdp.stateaction import MDPState
from mlpy.mdp.continuous import CbTData, CASML


def calc_stats(obs):
    nobs = obs.shape[0]
    d, n = obs[0].shape

    obs_avg = np.zeros((n, d))

    for i, o in enumerate(obs):
        obs_avg += o.T

    obs_avg /= float(nobs)

    return obs_avg


def plot_sampled(obs, sampled, fig1=None, fig2=None, ax1=None, ax2=None, ax3=None, ax4=None):
    if fig1 is None or not plt.fignum_exists(fig1.number):
        fig1 = plt.figure()
        fig2 = plt.figure()
        plt.rcParams['legend.fontsize'] = 10
        ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
        ax2 = fig2.add_subplot(1, 3, 1)
        ax3 = fig2.add_subplot(1, 3, 2)
        ax4 = fig2.add_subplot(1, 3, 3)
        fig1.show()
        fig2.show()

    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()

    for i, o in enumerate(obs):
        for t in o.T:
            [x, y, z] = t
            ax1.scatter(x, y, z, edgecolors='b', c='b', marker='o')
            ax2.scatter(x, y, edgecolors='b', c='b', marker='o')
            ax3.scatter(x, z, edgecolors='b', c='b', marker='o')
            ax4.scatter(y, z, edgecolors='b', c='b', marker='o')

        ax1.plot(o[0], o[1], o[2], c='b', label='trajectory {0}'.format(i + 1))
        ax2.plot(o[0], o[1], c='b', label='trajectory {0}'.format(i + 1))
        ax3.plot(o[0], o[2], c='b', label='trajectory {0}'.format(i + 1))
        ax4.plot(o[1], o[2], c='b', label='trajectory {0}'.format(i + 1))

    for s in sampled.T:
        ax1.scatter(s[0], s[1], s[2], edgecolors='y', c='y', marker='^')
        ax2.scatter(s[0], s[1], edgecolors='y', c='y', marker='^')
        ax3.scatter(s[0], s[2], edgecolors='y', c='y', marker='^')
        ax4.scatter(s[1], s[2], edgecolors='y', c='y', marker='^')

    ax1.plot(sampled[0], sampled[1], sampled[2], c='y', label='sampled')
    ax2.plot(sampled[0], sampled[1], c='y', label='sampled')
    ax3.plot(sampled[0], sampled[2], c='y', label='sampled')
    ax4.plot(sampled[1], sampled[2], c='y', label='sampled')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_zlabel('Z position')
    ax1.set_title('Failed: Observed and sampled trajectories')

    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    ax2.set_title('Observed and sampled trajectories: X, Y')

    ax3.set_xlabel('X position')
    ax3.set_ylabel('Z position')
    ax3.set_title('Observed and sampled trajectories: X, Z')

    ax4.set_xlabel('Y position')
    ax4.set_ylabel('Z position')
    ax4.set_title('Observed and sampled trajectories: Y, Z')

    fig1.canvas.draw()
    fig2.canvas.draw()

    return fig1, fig2, ax1, ax2, ax3, ax4


def main(args):
    try:
        data = load_from_file(args.infile)
        train = data["train"]
        test = data["test"]

        obs_avg = calc_stats(test)

        nobs = 20   # train[0].shape[0]
        d, n = train[0][0].shape
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

    ntrials = 20

    fig1 = None
    fig2 = None
    ax1 = None
    ax2 = None
    ax3 = None
    ax4 = None

    try:
        radius = [0.01, 0.02, 0.025]     # [0.001, 0.005, 0.01, 0.015]
        for i in xrange(nobs * len(radius)):

            if i % nobs == 0:
                param = radius[i]

            case_t_template = {
                "state": {
                    "type": "float",
                    "value": "data.state",
                    "is_index": True,
                    "retrieval_method": 'radius-n',
                    "retrieval_method_params": param,
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

            model = CASML(CbTData(case_t_template, rho=args.rho, tau=args.tau, sigma=args.sigma,
                                  plot_reuse=False, plot_reuse_params='original_origin'),
                          ncomponents=args.ncomponents)

            with Timer() as tm:
                for j, states in enumerate(train[i]):
                    # Train CASML's case base and hmm with states and actions
                    model.fit(states, actions)
            print('Model trained in %.03f sec.' % tm.time)

            failed = 0
            sampled = np.zeros((ntrials, d, n))
            ncases = (model._cb_t._counter - 1)

            for j in xrange(ntrials):
                with Timer() as tm:
                    # Test model
                    iter_ = 0
                    nfails = 0

                    while True:
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
                            nfails += 1

                            print "{0}:{1} Failed to infer next state distribution at step {2}.".format(j,
                                                                                                        nfails,
                                                                                                        iter_)
                            continue
                        break

                    failed += nfails

                print('Request took %.03f sec.' % tm.time)

            failed /= float(ntrials)

            sampled = np.sum(sampled, axis=0)
            sampled /= ntrials
            # fig1, fig2, ax1, ax2, ax3, ax4 = plot_sampled([obs_avg.T], sampled, fig1, fig2, ax1, ax2, ax3, ax4)

            filename, extension = os.path.splitext(args.outfile)
            save_to_file(filename + '_train_' + str(i) + '_radius_' + str(param) + extension, {
                "sampled": sampled,
                "failed": failed,
                "ncases": ncases,
            })

    except IOError:
        sys.exit(sys.exc_info()[1])
    except KeyError, e:
        sys.exit("Key not found: {0}".format(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment: Learn continuous action and state model for motion.")
    ap.add_argument("--rho", type=float, default=0.97, required=False, help="The maximum error rho.")
    ap.add_argument("--tau", type=float, default=0.005, required=False, help="The maximum error tau.")
    ap.add_argument("--sigma", type=float, default=0.001, required=False, help="The maximum error sigma.")
    ap.add_argument("--ncomponents", type=int, default=10, required=False, help="The number of hidden states.")
    ap.add_argument("--infile", type=str, required=True, help="The trajectory data file name.")
    ap.add_argument("--policy", type=str, required=True, help="The policy file name.")
    ap.add_argument("--policy_num", type=str, default=0, required=False,
                    help="The identification of the policy to run")
    ap.add_argument("--outfile", type=str, required=False, help="The collected data file name.")
    main(ap.parse_args())
