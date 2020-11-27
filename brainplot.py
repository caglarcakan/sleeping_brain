import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy import stats

import datetime

import networkx as nx
import matplotlib as mpl
import sys
import tqdm
from matplotlib import animation, rc
from matplotlib.ticker import MaxNLocator

from IPython.display import clear_output, HTML, Image


from neurolib.utils import atlases
import neurolib.utils.paths as paths
import neurolib.utils.functions as func


class Brainplot:
    def __init__(self, Cmat, data, nframes=None, dt=0.1, fps=25, labels=False, darkmode=True):
        self.sc = Cmat
        self.n = self.sc.shape[0]

        self.data = data
        self.darkmode = darkmode

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))

        coords = {}
        atlas = atlases.AutomatedAnatomicalParcellation2()
        for i, c in enumerate(atlas.coords()):
            coords[i] = [c[0], c[1]]
        self.position = coords

        self.edge_threshold = 0.01

        self.fps = fps
        self.dt = dt

        nframes = nframes or int((data.shape[1] * self.dt / 1000) * self.fps)  # 20 fps default
        logging.info(f"Defaulting to {nframes} frames at {self.fps} fp/s")
        self.nframes = nframes

        self.frame_interval = self.data.shape[1] // self.nframes

        self.interval = int(self.frame_interval * self.dt)

        self.draw_labels = labels

        for t in range(self.n):
            # print t
            for s in range(t):
                # print( n, t, s)
                if self.sc[t, s] > self.edge_threshold:
                    # print( 'edge', t, s, self.sc[t,s])
                    self.G.add_edge(t, s)

        # node color map
        self.cmap = plt.get_cmap("plasma")  # mpl.cm.cool

        # default style

        self.imagealpha = 0.5

        self.edgecolor = "k"
        self.edgealpha = 0.8
        self.edgeweight = 1.0

        self.nodesize = 50
        self.nodealpha = 0.8
        self.vmin = 0
        self.vmax = 50

        self.lw = 0.5

        if self.darkmode:
            plt.style.use("dark")
            # let's choose a cyberpunk style for the dark theme
            self.edgecolor = "#37f522"
            self.edgeweight = 0.5
            self.edgealpha = 0.6

            self.nodesize = 40
            self.nodealpha = 0.8
            self.vmin = 0
            self.vmax = 30
            self.cmap = plt.get_cmap("cool")  # mpl.cm.cool

            self.imagealpha = 0.5

            self.lw = 1

            fname = os.path.join("neurolib", "data", "resources", "clean_brain_white.png")
        else:
            # plt.style.use("light")
            fname = os.path.join("neurolib", "data", "resources", "clean_brain.png")

        self.imgTopView = mpl.image.imread(fname)

        self.pbar = tqdm.tqdm(total=self.nframes)

    def update(self, i, ax, ax_rates=None, node_color=None, node_size=None, node_alpha=None, clear=True):
        frame = int(i * self.frame_interval)

        node_color = node_color or self.data[:, frame]
        node_size = node_size or self.nodesize
        node_alpha = node_alpha or self.nodealpha
        if clear:
            ax.cla()
        im = ax.imshow(self.imgTopView, alpha=self.imagealpha, origin="upper", extent=[40, 202, 28, 240])
        ns = nx.draw_networkx_nodes(
            self.G,
            pos=self.position,
            node_color=node_color,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            node_size=node_size,
            alpha=node_alpha,
            ax=ax,
            edgecolors="k",
        )
        es = nx.draw_networkx_edges(
            self.G, pos=self.position, alpha=self.edgealpha, edge_color=self.edgecolor, ax=ax, width=self.edgeweight
        )

        labels = {}
        for ni in range(self.n):
            labels[ni] = str(ni)

        if self.draw_labels:
            nx.draw_networkx_labels(self.G, self.position, labels, font_size=8)

        ax.set_axis_off()
        ax.set_xlim(20, 222)
        ax.set_ylim(25, 245)

        # timeseries
        if ax_rates:
            ax_rates.cla()
            ax_rates.set_xticks([])
            ax_rates.set_yticks([])
            ax_rates.set_ylabel("Brain activity", fontsize=8)

            t = np.linspace(0, frame * self.dt, frame)
            ax_rates.plot(t, np.mean(self.data[:, :frame], axis=0).T, lw=self.lw)

            t_total = self.data.shape[1] * self.dt
            ax_rates.set_xlim(0, t_total)

        self.pbar.update(1)
        plt.tight_layout()
        if clear:
            clear_output(wait=True)


def plot_rates(model):
    plt.figure(figsize=(4, 1))
    plt_until = 10 * 1000
    plt.plot(model.t[model.t < plt_until], model.output[:, model.t < plt_until].T, lw=0.5)


def plot_brain(
    model, ds, color=None, size=None, title=None, cbar=True, cmap="RdBu", clim=None, cbarticks=None, cbarticklabels=None
):
    """Dump and easy wrapper around the brain plotting function.

    :param color: colors of nodes, defaults to None
    :type color: numpy.ndarray, optional
    :param size: size of the nodes, defaults to None
    :type size: numpy.ndarray, optional
    :raises ValueError: Raises error if node size is too big.
    """
    s = Brainplot(ds.Cmat, model.output, fps=10, darkmode=False)
    s.cmap = plt.get_cmap(cmap)

    dpi = 300
    fig = plt.figure(dpi=dpi)
    ax = plt.gca()
    if title:
        ax.set_title(title, fontsize=26)

    if clim is None:
        s.vmin, s.vmax = np.min(color), np.max(color)
    else:
        s.vmin, s.vmax = clim[0], clim[1]

    if size is not None:
        node_size = size
    else:
        # some weird scaling of the color to a size
        def norm(what):
            what = what.copy()
            what -= np.min(what)
            what /= np.max(what)
            return what

        node_size = list(np.exp((norm(color) + 2) * 2))

    if isinstance(color, np.ndarray):
        color = list(color)
    if isinstance(node_size, np.ndarray):
        node_size = list(node_size)

    if np.max(node_size) > 2000:
        raise ValueError(f"node_size too big: {np.max(node_size)}")
    s.update(0, ax, node_color=color, node_size=node_size, clear=False)
    if cbar:
        cbaxes = fig.add_axes([0.68, 0.1, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=s.cmap, norm=plt.Normalize(vmin=s.vmin, vmax=s.vmax))
        cbar = plt.colorbar(sm, cbaxes, ticks=cbarticks)
        cbar.ax.tick_params(labelsize=16)
        if cbarticklabels:
            cbar.ax.set_yticklabels(cbarticklabels)


# other plotting


def plot_average_timeseries(model, xticks=False, kwargs={}, xlim=None, figsize=(8, 1)):
    # print("{} Peaks found ({} p/s)".format(len(peaks), len(peaks)/(model.params['duration']/1000)))
    plt.figure(figsize=figsize, dpi=300)

    # cut rates if xlim is given
    if xlim:
        rates = model.output[:, (model.t / 1000 > xlim[0]) & (model.t / 1000 < xlim[1])]
        t = model.t[(model.t / 1000 > xlim[0]) & (model.t / 1000 < xlim[1])]
        print(rates.shape)
    else:
        rates = model.output
        t = model.t

    plt.plot(t / 1000, np.mean(rates, axis=0), **kwargs)
    # plt.plot(model.t/1000, signal, lw=2, label = 'smoothed')
    plt.autoscale(enable=True, axis="x", tight=True)

    # for p in peaks:
    #    plt.vlines(p*params['dt']*10, 0, 0.008, color='b', lw=2, alpha=0.6)
    # plt.xlim(0, 10000)
    # plt.plot(model.t/1000, states*10, lw=2, c='C1', alpha=0.4, label='detected state')
    # plt.xlabel("Time [s]")
    if not xticks:
        plt.xticks([])
    if xlim:
        plt.xlim(*xlim)

    plt.ylabel("Rate [Hz]")

    # plt.legend(loc=1, fontsize=8)

    # import matplotlib as mpl

    # mpl.rcParams['axes.spines.left'] = False
    # mpl.rcParams['axes.spines.right'] = False
    # mpl.rcParams['axes.spines.top'] = False
    # mpl.rcParams['axes.spines.bottom'] = False

    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.tight_layout()
    set_axis_size(*figsize)


def detectSWStates(output, threshold=1):
    states = np.zeros(output.shape)
    # super_threshold_indices = rates > threshold
    # works with 1D thresholds, per node
    super_threshold_indices = np.greater(output.T, threshold).T
    states[super_threshold_indices] = 1
    return states


def filter_long_states(model, states):
    states = states.copy()
    LENGTH_THRESHOLD = int(50 / model.params["dt"])  # ms
    for s in states:
        lengths = get_state_lengths(s)
        current_idx = 0
        last_long_state = lengths[0][0]  # first state
        for state, length in lengths:
            if length >= LENGTH_THRESHOLD:
                last_long_state = state
            else:
                s[current_idx : current_idx + length] = last_long_state
            current_idx += length
    return states


def detectSWs(model, up_thresh=0.01, detectSWStates_kw=None, filter_long=True):
    threshold_per_node = up_thresh * np.max(model.output, axis=1)  # as per Nghiem et al., 2020, Renart et al., 2010
    states = np.zeros(model.output.shape)
    if detectSWStates_kw:
        logging.warning(f"Warning: detectSWStates_kw is not implemented anymore, using up_thresh = {up_thresh}")

    states = detectSWStates(model.output, threshold=threshold_per_node)
    # for i, o in enumerate(model.output):
    #     smoothed = o  # scipy.ndimage.gaussian_filter(o, sigma=500)
    #     s = detectSWStates(model.t, smoothed, **detectSWStates_kw)
    #     states[i, :] = s
    if filter_long:
        states = filter_long_states(model, states)
    return states


def get_involvement(states, down=True):
    """Returns involvement in up- and down-states.

    :param states: state array (NxT)
    :type states: np.ndarray
    :return: Involvement time series (1xT)
    :rtype: np.ndarray
    """
    up_involvement = np.sum(states, axis=0) / states.shape[0]
    if down:
        return 1 - up_involvement
    else:
        return up_involvement


def plot_states_timeseries(model, states, title=None, labels=True, cmap="plasma"):
    figsize = (8, 2)
    plt.figure(figsize=figsize)
    plt.imshow(
        states,
        extent=[0, states.shape[1] * model.params.dt / 1000, 0, states.shape[0]],
        aspect="auto",
        cmap=plt.cm.get_cmap(cmap, 2),
    )
    if labels:
        plt.xlabel("Time [s]")
        plt.ylabel("Node")
    else:
        plt.xticks([])
        plt.yticks([])
    if title:
        plt.title(title)

    plt.autoscale(enable=True, axis="x", tight=True)

    if labels:
        cbar = plt.colorbar(pad=0.02)
        cbar.set_ticks([0.25, 0.75])
        cbar.ax.set_yticklabels(["Down", "Up"], rotation=90, va="center")
        cbar.ax.tick_params(width=0, labelsize=14)
    plt.tight_layout()
    set_axis_size(*figsize)


def plot_involvement_timeseries(model, involvement):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [4, 1]})
    axs[0].set_title("Involvement of brain areas in SO events")
    axs[0].plot(model.t / 1000, involvement * 100, c="C0")
    axs[0].set_ylabel("Involvement [%]")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylim([0, 100])
    axs[0].set_aspect("auto")

    axs[1].hist(involvement * 100, bins=10, orientation="horizontal", density=True, rwidth=0.8, edgecolor="k")
    axs[1].set_yticks([])
    axs[1].set_xlabel("KDE")
    axs[1].set_ylim([0, 100])
    plt.tight_layout()


def plot_degree_duration_scatterplot(model, states, ds, lingres=False, color_down="C0", color_up="C1"):
    figsize = (2, 2)
    plt.figure(figsize=figsize)
    area_downtimes = np.sum(states == 0, axis=1) / model.output.shape[1] * 100
    area_uptimes = np.sum(states == 1, axis=1) / model.output.shape[1] * 100
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    degrees = [(np.sum(ds.Cmat, axis=1))[i] / np.max(np.sum(ds.Cmat, axis=1)) for i in range(ds.Cmat.shape[0])]

    ax.scatter(
        degrees,
        area_uptimes,
        s=14,
        c=color_up,
        edgecolor="black",
        linewidth=0.5,
        label="up-state",
    )

    ax.scatter(
        degrees,
        area_downtimes,
        s=14,
        c=color_down,
        edgecolor="black",
        linewidth=0.5,
        label="down-state",
    )
    if lingres:
        plot_linregress(degrees, area_downtimes, kwargs={"c": color_down, "zorder": -2})

    if lingres:
        plot_linregress(degrees, area_uptimes, kwargs={"c": color_up, "zorder": -2})

    for i in range(ds.Cmat.shape[0]):
        degree = (np.sum(ds.Cmat, axis=1))[i] / np.max(np.sum(ds.Cmat, axis=1))

    plt.legend(fontsize=12, frameon=False, markerscale=1.8, handletextpad=-0.5)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Time spent [%]")
    plt.tight_layout()
    set_axis_size(*figsize)


def plot_transition_phases(node_mean_phases_down, node_mean_phases_up, atlas):
    def hide_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    names = []
    phases_down = []
    phases_up = []
    ipsilateral_mean_down_phase = [
        (i + k) / 2 for i, k in zip(node_mean_phases_down[0::2], node_mean_phases_down[1::2])
    ]
    ipsilateral_mean_up_phase = [(i + k) / 2 for i, k in zip(node_mean_phases_up[0::2], node_mean_phases_up[1::2])]
    # ipsilateral_names = [i[:-2] for i, k in zip(atlas.names()[0::2], atlas.names()[1::2])]
    ipsilateral_names = [
        f"{i[:-2]} ({nr*2}, {nr*2+1})" for nr, (i, k) in enumerate(zip(atlas.names()[0::2], atlas.names()[1::2]))
    ]
    # clean up names
    for i in range(len(ipsilateral_names)):
        ipsilateral_names[i] = ipsilateral_names[i].replace("_", " ")

    for i, ipsi_region in enumerate(np.argsort(ipsilateral_mean_down_phase)):
        # print(i, region, node_mean_phases_down[region], atlas.names()[region], atlas.coords()[region][2])
        # y_coord = (i%80)/30
        names.append(ipsilateral_names[ipsi_region])
        phases_down.append(ipsilateral_mean_down_phase[ipsi_region])
        phases_up.append(ipsilateral_mean_up_phase[ipsi_region])

    names = [n.replace("_", " ") for n in names]
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    # left=-np.asarray(phases[::-1]),
    ax.bar(names, phases_up, lw=1, ls="--", color="C0", label="up-transition")
    ax.bar(names, phases_down, lw=1, ls="--", color="C1", label="down-transition")
    # ax.tick_params(axis="both", which="major", labelsize=8)
    # ax.yaxis.set_tick_params(width=1)
    ax.autoscale(enable=True, axis="x", tight=True)

    ax.legend(fontsize=8, loc=1, bbox_to_anchor=(0.98, 1.0))
    ax.set_ylabel("Phase of transition", fontsize=12)
    # ax.text(0.2, 80, "Up-state", fontsize=10)
    # ax.text(-0.2, 80, "Down-state", fontsize=10, ha="right")
    # ax.set_title("Mean transition phase")

    hide_axis(ax)
    plt.grid(alpha=0.5, lw=0.5)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", which="major", labelsize=7, rotation=90)
    plt.tight_layout(pad=0)


def plot_transition_phases_against_eachother(node_mean_phases_down, node_mean_phases_up):
    figsize = (2, 2)
    figsize = np.multiply(figsize, 0.75)
    plt.figure(figsize=figsize)
    plt.scatter(
        node_mean_phases_down,
        node_mean_phases_up,
        s=8,
        edgecolor="black",
        linewidth=0.5,
        c="lightgray",
    )
    plot_linregress(node_mean_phases_down, node_mean_phases_up, kwargs={"c": "k", "alpha": 0.5, "zorder": -2})
    plt.xlabel("Mean down-phase")
    plt.ylabel("Up-phase")
    plt.tight_layout(pad=0)
    set_axis_size(*figsize)


def plot_state_brain_areas(model, states, atlas, ds, color_down="C0", color_up="C1"):
    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    # axs.set_title("Time spent in up- and down-state per brain area", fontsize=12)
    area_downtimes = np.sum(states == 0, axis=1) / model.output.shape[1] * 100
    area_uptimes = np.sum(states == 1, axis=1) / model.output.shape[1] * 100

    ipsilateral_downtimes = [(i + k) / 2 for i, k in zip(area_downtimes[0::2], area_downtimes[1::2])]
    ipsilateral_uptimes = [(i + k) / 2 for i, k in zip(area_uptimes[0::2], area_uptimes[1::2])]
    ipsilateral_uptimes_diff = [(i - k) for i, k in zip(area_uptimes[0::2], area_uptimes[1::2])]

    ipsilateral_names = [
        f"{i[:-2]} ({nr*2}, {nr*2+1})" for nr, (i, k) in enumerate(zip(atlas.names()[0::2], atlas.names()[1::2]))
    ]
    # clean up names
    for i in range(len(ipsilateral_names)):
        ipsilateral_names[i] = ipsilateral_names[i].replace("_", " ")

    axs.bar(
        ipsilateral_names,
        ipsilateral_uptimes,
        bottom=ipsilateral_downtimes,
        edgecolor="k",
        color="C0",
        linewidth="0.2",
        zorder=-10,
        label="up-state",
    )
    axs.bar(ipsilateral_names, ipsilateral_downtimes, edgecolor="k", color="C1", linewidth="0.2", label="down-state")
    axs.legend(fontsize=8, loc=1, bbox_to_anchor=(0.98, 0.9))

    axs.autoscale(enable=True, axis="x", tight=True)

    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    axs.spines["left"].set_visible(False)
    axs.spines["bottom"].set_visible(False)

    axs.tick_params(axis="x", which="major", labelsize=7, rotation=90)

    # axs[0].set_xlabel("Brain area")
    axs.set_ylabel("Time spent [%]", fontsize=12)
    axs.set_yticks([0, 25, 50, 75, 100])
    axs.tick_params(axis="y", labelsize=10)

    # for i in range(ds.Cmat.shape[0]):
    #     degree = (np.sum(ds.Cmat, axis=1))[i] / np.max(np.sum(ds.Cmat, axis=1))
    #     axs[1].scatter(degree, area_downtimes[i], s=5, c="C1", edgecolor="black", linewidth="0.2")
    #     axs[1].scatter(degree, area_uptimes[i], s=5, c="C0", edgecolor="black", linewidth="0.2")
    # axs[1].set_xlabel("Node degree")
    # axs[1].set_xticks([0, 1, 2, 3])
    # axs[1].set_yticklabels([])
    plt.tight_layout(pad=0)
    return area_downtimes, area_uptimes


def plot_involvement_durations(model, states, involvement, nbins=6, legend=False):
    invs = {0: [], 1: []}
    lens = {0: [], 1: []}
    for s in states[:]:
        lengths = get_state_lengths(s)
        current_idx = 0
        for state, length in lengths:
            state = int(state)
            # compuite average involvement during this state
            mean_involvement = np.mean(involvement[current_idx : current_idx + length])
            lens[state].append(length * model.params.dt)  # convert to seconds
            invs[state].append(mean_involvement)

            current_idx += length

    from scipy.stats import binned_statistic

    figsize = (3, 2)
    figsize = np.multiply(figsize, 0.75)
    plt.figure(figsize=figsize)
    up_bin_means, bin_edges, _ = binned_statistic(invs[1], lens[1], bins=nbins, range=(0, 1))
    plt.bar(bin_edges[:-1], up_bin_means[::-1], width=0.04, edgecolor="k", color="C0", label="up-state")

    down_bin_means, bin_edges, _ = binned_statistic(invs[0], lens[0], bins=nbins, range=(0, 1))
    plt.bar(bin_edges[:-1] + 0.05, down_bin_means, width=0.04, edgecolor="k", color="C1", label="down-state")
    if legend:
        plt.legend(fontsize=8)

    plt.xticks([0, 0.5, 1], [0, 50, 100])
    axs = plt.gca()
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    axs.spines["left"].set_visible(False)
    plt.gca().tick_params(axis="both", direction="out", which="major", bottom=True, left=True)
    plt.xlabel("Involvement [%]")
    plt.ylabel("Duration [ms]")
    plt.tight_layout()
    set_axis_size(*figsize)


def get_state_durations_flat(model, states):
    durations = get_state_lengths(states)
    ups, downs = get_updown_lengths(durations)

    dt_to_s = model.params.dt / 1000  # return seconds
    flat_ups = [u * dt_to_s for up in ups for u in up]
    flat_downs = [d * dt_to_s for down in downs for d in down]
    return flat_ups, flat_downs


def plot_state_durations(model, states, legend=True, alpha=1.0):
    durations = get_state_lengths(states)
    ups, downs = get_updown_lengths(durations)

    dt_to_s = model.params.dt / 1000
    flat_ups = [u * dt_to_s for up in ups for u in up]
    flat_downs = [d * dt_to_s for down in downs for d in down]
    # figsize = (3, 2.0)
    figsize = (2, 1.7)
    # figsize = np.multiply(figsize, 0.75)
    plt.figure(figsize=figsize)
    plt.hist(flat_ups, color="C0", label="up", edgecolor="k", linewidth=0.75)
    plt.hist(flat_downs, color="C1", label="down", edgecolor="k", linewidth=0.75, alpha=alpha)
    if legend:
        plt.legend(fontsize=12, frameon=False)
    plt.gca().set_yscale("log")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.ylabel("Log-probability")
    plt.yticks([])
    plt.xlabel("Duration [s]")
    plt.gca().tick_params(axis="x", direction="out", which="major", bottom=True, left=True)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    set_axis_size(*figsize)


def plot_involvement_distribution(model, involvement, remove_xticks=True, color_localglobal=False):
    # figsize = (3, 2)
    figsize = (2, 1.7)
    # figsize = np.multiply(figsize, 0.75)
    plt.figure(figsize=figsize)
    N, bins, patches = plt.hist(involvement * 100, bins=12, density=True, rwidth=0.8, edgecolor="k", color="C1")

    if color_localglobal:
        for i in range(0, len(patches) // 2):
            patches[i].set_facecolor("C0")
        for i in range(len(patches) // 2, len(patches)):
            patches[i].set_facecolor("C1")

    plt.xticks([0, 50, 100])
    if remove_xticks:
        plt.yticks([])
    else:
        y_vals = plt.gca().get_yticks()
        plt.gca().set_yticklabels(["{:3.0f}".format(x * 100) for x in y_vals])
    plt.ylabel("Probability")
    plt.xlabel("Involvement [%]")
    plt.gca().tick_params(axis="x", direction="out", which="major", bottom=True, left=True)
    plt.xlim([0, 100])
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    max_y = plt.gca().get_ylim()[1] * 1.1
    plt.ylim(0, max_y * 1.1)
    # plt.vlines([50], 0, max_y, linestyles="--", colors="#363638", alpha=0.9)
    plt.text(25, max_y * 0.95, "local", fontsize=14, color="C0", ha="center")
    plt.text(75, max_y * 0.95, "global", fontsize=14, color="C1", ha="center")
    plt.tight_layout()
    set_axis_size(*figsize)


def plot_involvement_mean_amplitude(model, involvement, skip=300, lingress=True):
    figsize = (3, 2)
    figsize = np.multiply(figsize, 0.75)
    plt.figure(figsize=figsize)
    rates = np.mean(model.output, axis=0)[::skip]
    plt.scatter(
        involvement[::skip] * 100,
        rates,
        # scipy.stats.zscore(np.mean(model.output, axis=0)[::skip]),
        s=10,
        edgecolor="w",
        linewidth=0.2,
        c="C1",
        alpha=0.7,
    )
    # plt.gca().axhline(0, linestyle="--", lw=1, color="#ABB2BF", zorder=-1)
    plt.xlim(0, 100)
    plt.xticks([0, 50, 100])
    plt.xlabel("Involvement [%]")
    plt.ylabel("Rate [Hz]")

    if lingress:
        slope, intercept, r_value, p_value, std_err = plot_linregress(
            involvement[::skip] * 100, rates, kwargs={"c": "k", "alpha": 0.75}
        )
        plt.text(50, np.max(rates) * 0.75, f"$R^2={r_value**2:0.2f}$")

    axs = plt.gca()
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    axs.spines["left"].set_visible(False)
    axs.spines["bottom"].set_visible(False)
    plt.gca().tick_params(axis="both", direction="out", which="major", bottom=True, left=True)

    plt.tight_layout()
    set_axis_size(*figsize)


def get_state_lengths(xs):
    """
    Get's the length of successive elements in a list.
    Useful for computing inter-spike-intervals (ISI) or the length
    of states.

    Example: get_state_lengths([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0])
    Returns: [(0, 4), (1, 4), (0, 2), (1, 1), (0, 1)]

    :param xs: Input list with successive states
    :type xs: list
    :return: List of (state, length) tuples
    :rtype: list
    """
    import itertools

    if np.array(xs).ndim == 1:
        durations = [(x, len(list(y))) for x, y in itertools.groupby(xs)]
    elif np.array(xs).ndim > 1:
        durations = [get_state_lengths(xss) for xss in xs]
    return durations


def get_updown_lengths(durations):
    """Returns the length of all up- and down-states for each node"""
    ups = []
    downs = []
    for i, l in enumerate(durations):
        ups.append([u[1] for u in l if u[0] == 1.0])
        downs.append([u[1] for u in l if u[0] == 0.0])
    return ups, downs


def plot_linregress(x, y, plot=True, kwargs={}):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"slope = {slope}, intercept = {intercept}, r_value = {r_value}, p_value = {p_value}, std_err = {std_err}")
    if plot:
        plt.plot(x, intercept + x * slope, **kwargs)
    return slope, intercept, r_value, p_value, std_err


def plot_down_up_durations(model, durations, ds, plot_regression=False):
    import matplotlib.cm as cm

    all_ups = []
    all_downs = []
    all_colors = []
    for n, dur in enumerate(durations):
        # make sure to start with a down-state
        skip = 0 if dur[0][0] == 0.0 else 1
        down_durations = [d[1] * model.params.dt / 1000 for d in dur[skip::2]]
        up_durations = [d[1] * model.params.dt / 1000 for d in dur[skip + 1 :: 2]]
        # normalized degree [0, 1]
        degree = (np.sum(ds.Cmat, axis=1))[n] / np.max(np.sum(ds.Cmat, axis=1))
        for d, u in zip(down_durations, up_durations):
            all_ups.append(u)
            all_downs.append(d)
            all_colors.append(degree)

    fig_scale = 1.2
    plt.figure(figsize=(3 * fig_scale, 2.5 * fig_scale))
    plt.scatter(all_downs, all_ups, s=2, c=all_colors, cmap="plasma", alpha=0.8)
    plt.ylabel("Next up-state duration [s]")
    plt.xlabel("Down-state duration [s]")

    plot_linregress(all_downs, all_ups, plot=plot_regression)

    cbar = plt.colorbar()
    cbar.set_label(label="Node degree", size=10)
    cbar.ax.tick_params(labelsize=10)
    plt.show()


def get_transition_phases(states, phase):
    transitions = np.diff(states)
    node_mean_phases_down = []
    node_mean_phases_up = []

    for ni, trs in enumerate(transitions):
        down_phases = []
        up_phases = []
        for i, t in enumerate(trs):
            if t == -1:
                down_phases.append(float(phase[i]))
            if t == 1:
                up_phases.append(float(phase[i]))
        mean_down_phase = scipy.stats.circmean(down_phases, high=np.pi, low=-np.pi, nan_policy="raise")
        mean_up_phase = scipy.stats.circmean(up_phases, high=np.pi, low=-np.pi, nan_policy="raise")

        print(f"Node {ni}: mean_down_phase = {mean_down_phase}, mean_up_phase = {mean_up_phase}")

        node_mean_phases_down.append(mean_down_phase)
        node_mean_phases_up.append(mean_up_phase)

    return node_mean_phases_down, node_mean_phases_up


def phase_transition_coordinate(node_mean_phases_down, node_mean_phases_up, atlas):
    coords = atlas.coords()
    minmax_coords = [int(np.min([c[1] for c in coords])), int(np.max([c[1] for c in coords]))]

    figsize = (2, 2)
    plt.figure(figsize=figsize)
    # ATTENTION!!!! MULTIPLYING COORDINATES WITH -1 SO THE PLOT GOES ANTERIOR TO POSTERIOR
    y_coords = np.array([c[1] for c in coords]) * -1
    plt.scatter(y_coords, node_mean_phases_down, c="C1", s=14, edgecolor="k", linewidth=0.5, label="down")
    plt.scatter(y_coords, node_mean_phases_up, c="C0", s=14, edgecolor="k", linewidth=0.5, label="up")

    # plt.legend(fontsize=8)

    down_slope, down_intercept, down_r_value, down_p_value, down_std_err = stats.linregress(
        y_coords, node_mean_phases_down
    )
    print(f"Down transitions: slope: {down_slope} r: {down_r_value}, p: {down_p_value}")
    plt.plot(y_coords, (lambda c: down_intercept + down_slope * c)(y_coords), c="C1", label="x", zorder=-5)

    up_slope, up_intercept, up_r_value, up_p_value, up_std_err = stats.linregress(y_coords, node_mean_phases_up)
    print(f"Up transitions: slope: {up_slope} r: {up_r_value}, p: {up_p_value}")
    plt.plot(y_coords, (lambda c: up_intercept + up_slope * c)(y_coords), c="C0", label="x", zorder=-5)

    plt.xlabel("Coordinate")
    plt.xticks([np.min(y_coords) + 30, np.max(y_coords) - 30], [f"Anterior", f"Posterior"])
    plt.ylabel("Transition phase $\phi$")
    plt.tight_layout()
    set_axis_size(*figsize)

    return (
        down_slope,
        down_intercept,
        down_r_value,
        down_p_value,
        down_std_err,
        up_slope,
        up_intercept,
        up_r_value,
        up_p_value,
        up_std_err,
    )


def kuramoto(events):
    import tqdm

    # fill in event at the end of the timeseries
    events[:, -1] = True
    phases = []
    logging.info("Determining phases ...")
    for n, ev in tqdm.tqdm(enumerate(events), total=len(events)):
        maximalist = np.where(ev)[0]
        phases.append([])
        last_event = 0
        for m in maximalist:
            for t in range(last_event, m):
                phi = 2 * np.pi * float(t - last_event) / float(m - last_event)
                phases[n].append(phi)
            last_event = m
        phases[n].append(2 * np.pi)

    logging.info("Computing Kuramoto order parameter ...")
    # determine kuramoto order paramter
    kuramoto = []
    nTraces = events.shape[0]
    for t in tqdm.tqdm(range(events.shape[1]), total=events.shape[1]):
        R = 1j * 0
        for n in range(nTraces):
            R += np.exp(1j * phases[n][t])
        R /= nTraces
        kuramoto.append(np.absolute(R))
    return kuramoto


def plot_phases_and_involvement(model, involvement, phases, states):

    fig, axs = plt.subplots(2, 1, figsize=(4, 2), sharex=True)

    up_transitions = np.diff(states) > 0
    down_transitions = np.diff(states) < 0
    for i, ups in enumerate(up_transitions):
        axs[0].vlines(np.where(ups)[0] * model.params.dt / 1000, 0, 3, lw=0.2, alpha=1, color="C0")
    for i, downs in enumerate(down_transitions):
        axs[0].vlines(np.where(downs)[0] * model.params.dt / 1000, -3, 0, lw=0.2, alpha=1, color="C1")
    # annotate vlines
    axs[0].text(7.7, 7.0, "up-transitions", fontsize=8)
    axs[0].vlines(7.6, 6.8, 9.0, lw=1, color="C0")
    axs[0].text(7.7, 4.4, "down-transitions", fontsize=8)
    axs[0].vlines(7.6, 4.0, 6.2, lw=1, color="C1")

    # axs[0].fill_between(model.t/1000, phases, 0, where=phases<0, zorder=-2, color='C0', alpha=0.8, label='$\phi < 0$')
    # axs[0].fill_between(model.t/1000, phases, 0, where=phases>0, zorder=-2, color='C1', alpha=0.8, label='$\phi > 0$')

    axs[0].plot(model.t / 1000, phases, zorder=2, lw=1, label="Global phase $\phi$", c="fuchsia")
    axs[0].set_yticks([])
    axs[0].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=1)

    axs[1].plot(model.t / 1000, np.mean(model.output, axis=0), c="k", alpha=1, lw=1, label="Mean rate")
    axs[1].plot(model.t / 1000, involvement * 30, lw=1, c="C3", label="Involvement")

    axs[1].set_xlim(0, 10)
    axs[1].set_yticks([])
    axs[1].set_xticks([0, 5, 10])
    axs[1].tick_params(labelsize=8)
    axs[1].set_xlabel("Time [s]", fontsize=8)
    axs[1].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)

    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)


# used for RESULT-dynamical-regimes-EXP-5.ipynb
def plot_ts(
    model,
    plot_nodes=[0],
    plot_alpha=1,
    plot_mean_background=False,
    mean_alpha=0.3,
    shift_mean=False,
    adaptation=False,
    stimulus=False,
    stimulus_scale=1.0,
    log_power=False,
    log_freq=False,
    norm_power=True,
    title_add="",
    lw=1,
    figsize=(4, 1.5),
    labels=False,
    crop_zero=None,
    xlim=None,
    savename=None,
    fontsize=None,
    n_ticks=2,
    tick_width=1,
    tick_length=3,
    legend=False,
    yticks=None,
):

    _, axs = plt.subplots(1, 1, dpi=600, figsize=figsize)
    title = " ".join([f"{p} = {model.params[p]}" for p in ["mue_ext_mean", "mui_ext_mean", "b", "sigma_ou"]])
    title += " " + title_add

    plot_col = "k"
    if str(plot_nodes) == "all":
        plot_output = model.output
        plot_adaptation = model.outputs.IA
        plot_col = None
    elif str(plot_nodes) == "mean":
        plot_output = np.mean(model.output, axis=0)
        plot_adaptation = np.mean(model.outputs.IA, axis=0)
    else:
        plot_output = model.output[plot_nodes]
        plot_adaptation = model.outputs.IA[plot_nodes]
        plot_col = None if len(plot_nodes) > 1 else "k"

    if plot_mean_background:
        mean_signal = np.mean(model.output, axis=0)
    # plot_adaptation = model.outputs.IA if str(plot_nodes) == "all" else model.outputs.IA[plot_nodes]

    plot_t = model.t / 1000
    plot_stimulus = model.params["ext_exc_current"] * stimulus_scale

    if xlim:
        plot_t = plot_t[xlim[0] : xlim[1]]
        if plot_output.ndim == 2:
            plot_output = plot_output[:, xlim[0] : xlim[1]]
            if adaptation:
                plot_adaptation = plot_adaptation[:, xlim[0] : xlim[1]]
        else:
            plot_output = plot_output[xlim[0] : xlim[1]]
            if adaptation:
                plot_adaptation = plot_adaptation[xlim[0] : xlim[1]]

        if stimulus and model.params["ext_exc_current"].ndim > 0:
            plot_stimulus = plot_stimulus[xlim[0] : xlim[1]]

        if plot_mean_background:
            mean_signal = mean_signal[xlim[0] : xlim[1]]

    axs.plot(
        plot_t,
        plot_output.T,
        alpha=plot_alpha,
        lw=lw,
        c=plot_col,
        # label="Firing rate" if len(plot_nodes) == 1 else None,
    )

    # plot the mean signal in the background?
    if plot_mean_background:
        if shift_mean:
            mean_signal /= np.max(mean_signal)
            mean_signal *= np.abs(np.max(plot_output) - np.min(plot_output)) / 4
            mean_signal = mean_signal + np.max(plot_output) * 1.1
        axs.plot(plot_t, mean_signal, alpha=mean_alpha, lw=lw, c="k", zorder=10, label="Mean brain")

    # if a stimulus was present
    if legend and stimulus and model.params["ext_exc_current"].ndim > 0:
        axs.plot(plot_t, plot_stimulus, lw=lw, c="C1", label="Stimulus", alpha=0.8)
        # add a line for adaptation current in legend
        if adaptation:
            axs.plot([0], [0], lw=lw, c="C0", label="Adaptation")
        leg = axs.legend(frameon=True, fontsize=14, framealpha=0.9)

        # Get the bounding box of the original legend
        bb = leg.get_bbox_to_anchor().inverse_transformed(axs.transAxes)

        # Change to location of the legend.
        yOffset = +0.0
        bb.y0 += yOffset
        bb.y1 += yOffset
        leg.set_bbox_to_anchor(bb, transform=axs.transAxes)

    if adaptation and not stimulus:
        color = "C0"
        ax_adapt = axs.twinx()
        # ax_adapt.set_ylabel("$I_A$ [pA]", color=color)
        ax_adapt.tick_params(
            axis="y",
            color=color,
            labelcolor=color,
            direction="out",
            length=tick_length,
            width=tick_width,
            labelsize=fontsize,
            right=False,
        )
        # plot_adaptation = model.outputs.IA if str(plot_nodes) == "all" else model.outputs.IA[plot_nodes]
        ax_adapt.plot(plot_t, plot_adaptation.T, lw=lw, label="Adaptation", color=color)
        # ax_adapt.legend(loc=4)
        if legend and adaptation:
            ax_adapt.legend(loc="lower right", frameon=True, fontsize=14, framealpha=0.9)

        ax_adapt.spines["right"].set_visible(False)
        ax_adapt.spines["top"].set_visible(False)
        ax_adapt.spines["left"].set_visible(False)
        ax_adapt.spines["bottom"].set_visible(False)
        ax_adapt.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))

    if labels:
        axs.set_xlabel("Time [s]")
        axs.set_ylabel("Rate $r_E$ [Hz]")
        if adaptation:
            ax_adapt.set_ylabel("Adaptation $I_A$ [pA]", color=color)

    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    axs.spines["left"].set_visible(False)
    axs.spines["bottom"].set_visible(False)

    # reduce number of ticks
    axs.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    if yticks is not None:
        axs.set_yticks(yticks)

    axs.tick_params(
        axis="both",
        direction="out",
        length=tick_length,
        width=tick_width,
        colors="k",
        labelsize=fontsize,
        bottom=False,
        left=False,
    )

    if crop_zero:
        axs.set_xlim(crop_zero, model.t[-1] / 1000)

    plt.tight_layout()

    # set the axis size precisely
    set_axis_size(figsize[0], figsize[1])

    if savename:
        save_fname = os.path.join(paths.FIGURES_DIR, f"{savename}")
        plt.savefig(save_fname)


# helper function
def set_axis_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_matrix(mat, cbarlabel, cbar=True, ylabels=True, plotlog=False):

    figsize = (2, 2)
    fig = plt.figure(figsize=figsize)
    # fc_fit = du.model_fit(model, ds, bold_transient=bold_transient, fc=True)["mean_fc_score"]
    # plt.title(f"FC (corr: {fc_fit:0.2f})", fontsize=12)
    if plotlog:
        from matplotlib.colors import LogNorm

        im = plt.imshow(mat, norm=LogNorm(vmin=10e-5, vmax=np.max(mat)), origin="upper")
    else:
        im = plt.imshow(mat, origin="upper")
    plt.ylabel("Node")
    plt.xlabel("Node")
    plt.xticks([0, 20, 40, 60])
    plt.yticks([0, 20, 40, 60])
    if ylabels == False:
        plt.ylabel("")
        plt.yticks([])

    if cbar:
        # cbaxes = fig.add_axes([0.95, 0.36, 0.02, 0.52])
        cbar = plt.colorbar(im, ax=plt.gca(), fraction=0.046, pad=0.04)
        # cbar = plt.colorbar(im, cbaxes)
        # cbar.set_ticks([0, 1])
        # cbar.ax.tick_params(width=0, labelsize=10)
        cbar.set_label(label=cbarlabel, size=10, labelpad=-1)
    plt.tight_layout()
    set_axis_size(*figsize)


################### plot fits
def plot_fc(bold, model=None, bold_transient=0, cbar=False, ylabels=True):
    if bold_transient and model:
        t_bold = model.outputs.BOLD.t_BOLD[model.outputs.BOLD.t_BOLD > bold_transient] / 1000
        bold = model.outputs.BOLD.BOLD[:, model.outputs.BOLD.t_BOLD > bold_transient]
    figsize = (2, 2)
    fig = plt.figure(figsize=figsize)
    # fc_fit = du.model_fit(model, ds, bold_transient=bold_transient, fc=True)["mean_fc_score"]
    # plt.title(f"FC (corr: {fc_fit:0.2f})", fontsize=12)
    im = plt.imshow(func.fc(bold), origin="upper", clim=(0, 1))
    plt.ylabel("Node")
    plt.xlabel("Node")
    plt.xticks([0, 20, 40, 60])
    plt.yticks([0, 20, 40, 60])

    if ylabels == False:
        plt.ylabel("")
        plt.yticks([])

    if cbar:
        cbaxes = fig.add_axes([0.95, 0.36, 0.02, 0.52])
        cbar = plt.colorbar(im, cbaxes)
        cbar.set_ticks([0, 1])
        cbar.ax.tick_params(width=0, labelsize=10)
        cbar.set_label(label="Correlation", size=10, labelpad=-1)
    plt.tight_layout()
    set_axis_size(*figsize)


def plot_fcd(bold, model=None, bold_transient=0, cbar=False, ylabels=True):
    if bold_transient and model:
        t_bold = model.outputs.BOLD.t_BOLD[model.outputs.BOLD.t_BOLD > bold_transient] / 1000
        bold = model.outputs.BOLD.BOLD[:, model.outputs.BOLD.t_BOLD > bold_transient]
    figsize = (2, 2)
    fig = plt.figure(figsize=figsize)
    fcd_matrix = func.fcd(bold)
    print(fcd_matrix.shape)
    im = plt.imshow(fcd_matrix, origin="upper")
    plt.xlabel("Time [min]")
    plt.ylabel("Time [min]")
    # nticks = 3
    # plt.xticks(np.linspace(0, fcd_matrix.shape[0] - 1, nticks), np.linspace(0, bold.shape[1] * 2.0, nticks, dtype=int))
    # plt.yticks(np.linspace(0, fcd_matrix.shape[0] - 1, nticks), np.linspace(0, bold.shape[1] * 2.0, nticks, dtype=int))
    plt.xticks([0, 32, 64], [0, 6, 12])
    plt.yticks([0, 32, 64], [0, 6, 12])

    if ylabels == False:
        plt.ylabel("")
        plt.yticks([])

    if cbar:
        cbaxes = fig.add_axes([0.9, 0.36, 0.02, 0.52])
        cbar = plt.colorbar(im, cbaxes)
        # cbar.set_ticks([0.4, 1])
        cbar.ax.locator_params(nbins=5)
        cbar.ax.tick_params(width=0, labelsize=10)
        cbar.set_label(label="Correlation", size=10)

    plt.tight_layout(w_pad=4.5)
    set_axis_size(*figsize)


def plot_fcd_distribution(model, ds, bold_transient=0):
    if bold_transient and model:
        t_bold = model.outputs.BOLD.t_BOLD[model.outputs.BOLD.t_BOLD > bold_transient] / 1000
        bold = model.outputs.BOLD.BOLD[:, model.outputs.BOLD.t_BOLD > bold_transient]
    figsize = (4, 3)
    fig = plt.figure(figsize=figsize)
    # plot distribution in fcd
    # axs[2, 1].set_title(f"FCD distance {fcd_fit:0.2f}", fontsize=12)
    plt.ylabel("Density")
    plt.yticks([])
    plt.xticks([0, 0.5, 1])
    plt.xlabel("FCD$_{ij}$")
    m1 = func.fcd(bold)
    triu_m1_vals = m1[np.triu_indices(m1.shape[0], k=1)]
    plt.hist(triu_m1_vals, density=True, color="springgreen", zorder=10, alpha=0.6, edgecolor="k")
    # plot fcd distributions of data
    if hasattr(ds, "FCDs"):
        for emp_fcd in ds.FCDs:
            m1 = emp_fcd
            triu_m1_vals = m1[np.triu_indices(m1.shape[0], k=1)]
            plt.hist(triu_m1_vals, density=True, alpha=0.5)
    plt.tight_layout()
    # set_axis_size(*figsize)


def plot_fc_over_time(model, ds, bold_transient=0):
    if bold_transient and model:
        t_bold = model.outputs.BOLD.t_BOLD[model.outputs.BOLD.t_BOLD > bold_transient] / 1000
        bold = model.outputs.BOLD.BOLD[:, model.outputs.BOLD.t_BOLD > bold_transient]
    # plt.title("FC corr over time", fontsize=12)
    figsize = (4, 3)
    fig = plt.figure(figsize=figsize)
    plt.plot(
        np.arange(4, bold.shape[1] * 2, step=2),
        np.array(
            [[func.matrix_correlation(func.fc(bold[:, :t]), fc) for t in range(2, bold.shape[1])] for fc in ds.FCs]
        ).T,
    )
    plt.ylabel("FC correlation")
    plt.xlabel("Simulation time [s]")
    plt.tight_layout()


def barplot_annotate_brackets(
    num1, num2, data, center, height, yerr=None, dh=0.05, barh=0.05, fs=None, maxasterix=None
):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ""
        p = 0.05

        while data < p:
            text += "*"
            p /= 10.0

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = "n. s."

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= ax_y1 - ax_y0
    barh *= ax_y1 - ax_y0

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    plt.plot(barx, bary, c="black")

    kwargs = dict(ha="center", va="bottom")
    if fs is not None:
        kwargs["fontsize"] = fs

    plt.text(*mid, text, **kwargs)


def plot_powerspectra(f, powerspecs, mean_power, savename=None, logx=False, logy=True, maxfr=None):
    """Plottnig EEG power spectra (RESULT-data-eeg-powerspectrum.ipynb)"""
    plt.figure(figsize=(3, 3), dpi=300)

    # plt.title(f"Sleep stage: {stage}")

    plt.plot(f, mean_power, c="k", zorder=5)

    if logy:
        plt.yscale("log", basey=2)
    if logx:
        plt.xscale("log", basex=2)

    for power in powerspecs:
        plt.plot(f, power, lw=1)
    plt.xlim(-1, 15)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Log power" if logy else "Power")
    plt.yticks([])
    # plt.xticks(np.linspace(0, 40, 81), fontsize=5)
    # plt.xlim(0, 5)

    if maxfr is None:
        maxfr = f[-1] + f[1]

    if not logx:
        plt.xticks(np.linspace(0, maxfr, 5), fontsize=12)

    plt.xlim(-0.5, maxfr + 0.5)

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    if savename:
        save_fname = os.path.join(paths.FIGURES_DIR, f"{savename}")
        plt.savefig(save_fname)

    # plt.show()
