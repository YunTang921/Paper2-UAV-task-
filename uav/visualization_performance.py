# 作用: 读取 Data/*.pkl 结果并生成性能/时间/奖励可视化; 依赖: baseline_analyze.py 产出的结果文件; 被依赖: 无
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset

# Returns tuple of handles, labels for axis ax, after reordering them to conform to the label order `order`,
# and if unique is True, after removing entries with duplicate labels.
def reorderLegend(ax=None, order=None, unique=False):
    if ax is None: ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))  # sort both labels and handles by labels
    if order is not None:  # Sort according to a given list (not necessarily complete)
        keys = dict(zip(order, range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t, keys=keys: keys.get(t[0], np.inf)))
    if unique:  labels, handles = zip(*unique_everseen(zip(labels, handles), key=labels))  # Keep only the first of
    # each handle
    ax.legend(handles, labels)
    return (handles, labels)


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x, k in zip(seq, key) if not (k in seen or seen_add(k))]


def cost_baseline_and_model_performance(verbose=False, save=False):
    File_paths = []
    type1 = "Data/Balanced_type1_2000_777.pkl"
    File_paths.append(type1)
    type2 = "Data/Balanced_type2_2000_777.pkl"
    File_paths.append(type2)
    greedy = "Data/Balanced_greedy_2000_777.pkl"
    File_paths.append(greedy)

    baseline_data = []
    for file_path in File_paths:
        with open(file_path, 'rb') as f:
            baseline_data.append(pickle.load(f))

    model_data_paths = []
    o_hetnet = "Data/Balanced_o_hetnet_2000_777.pkl"
    model_data_paths.append(o_hetnet)
    hetnet = "Data/Balanced_hetnet_2000_777.pkl"
    model_data_paths.append(hetnet)
    ptr = "Data/Balanced_ptr_2000_777.pkl"
    model_data_paths.append(ptr)


    model_data = []
    for model_data_path in model_data_paths:
        with open(model_data_path, 'rb') as f:
            model_data.append(pickle.load(f))

    fig, ax = plt.subplots(figsize=(12,8))
    # # COST
    colors = mcolors.TABLEAU_COLORS
    color_name = list(mcolors.TABLEAU_COLORS)

    print("----------PERFORMANCE--------------")

    ax.plot(baseline_data[1]['data'][0,:],baseline_data[1]['data'][4,:],linewidth=2,color=colors[color_name[1]],linestyle="-.", marker='^',markersize=7 ,label="OR-Type2")
    print("OR-Type2:",baseline_data[1]['data'][4,:])

    ax.plot(baseline_data[0]['data'][0,:],baseline_data[0]['data'][4,:],linewidth=2,color=colors[color_name[0]],linestyle="-.", marker='x',markersize=7 ,label="OR-Type1")
    print("OR-Type1:",baseline_data[0]['data'][4,:])

    ax.plot(baseline_data[2]['data'][0,:],baseline_data[2]['data'][4,:],linewidth=2,color=colors[color_name[2]],linestyle="-.", marker='s',markersize=4 ,label="Greedy")
    print("GREEDY:",baseline_data[2]['data'][4,:])
    ax.set_ylabel('Cost', fontsize=15, fontweight='bold')
    ax.set_xlabel('# of missions', fontsize=15, fontweight='bold')
    ax.set_xticks(baseline_data[0]['data'][0,:])
    ax.set_xticklabels((baseline_data[0]['data'][0,:]*3).astype(int))

    ax.plot(baseline_data[0]['data'][0,:], model_data[0]['data'][3],linewidth=2 ,marker='+',markersize=7 , color='y',
            label="TransDA-RL")

    ax.plot(baseline_data[0]['data'][0,:], model_data[1]['data'][3],linewidth=2 ,marker='o',markersize=5 , color='r',
            label="Transformer-RL")

    ax.plot(baseline_data[0]['data'][0,:], model_data[2]['data'][3], linewidth=2, marker='d', markersize=7, color='b',
            label="PointerNet-RL")



    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=15)

    ax.grid()
    fig.savefig('./Cost_Analyze.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 5))
    # TIME
    colors = mcolors.TABLEAU_COLORS
    color_name = list(mcolors.TABLEAU_COLORS)
    print("----------TIME--------------")

    ax.plot(baseline_data[0]['data'][0, :], baseline_data[0]['data'][4, :], linewidth=2, color=colors[color_name[0]],
            linestyle="-.", marker='x', markersize=7, label="OR-Tools")
    print("OR-Tools:", baseline_data[0]['data'][4, :])

    ax.plot(baseline_data[2]['data'][0, :], baseline_data[2]['data'][4, :], linewidth=2, color=colors[color_name[2]],
            linestyle="-.", marker='s', markersize=4, label="Greedy")
    print("GREEDY:", baseline_data[2]['data'][4, :])

    ax.plot(baseline_data[0]['data'][0, :], model_data[0]['data'][4], linewidth=2, marker='+', markersize=7, color='y',
            label="Attention-DARL")
    print("PointerNet:", model_data[1]['data'][4, :])

    ax.plot(baseline_data[0]['data'][0, :], model_data[1]['data'][4], linewidth=2, marker='o', markersize=5, color='r',
            label="Attention-RL")
    print("Ours:", model_data[0]['data'][4, :])

    ax.plot(baseline_data[0]['data'][0, :], model_data[2]['data'][4], linewidth=2, marker='o', markersize=5, color='b',
            label="Pointer-RL")
    print("Ours:", model_data[0]['data'][4, :])


    ax.set_ylabel('Time [s]', fontsize=15, fontweight='bold')
    ax.set_xlabel('# of missions', fontsize=15, fontweight='bold')
    ax.set_xticks(baseline_data[0]['data'][0, :])
    ax.set_xticklabels((baseline_data[0]['data'][0, :] * 3).astype(int))
    # ax.legend()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=8)

    ax.grid()
    # fig.savefig('./Archive/Time_Analyze.png', dpi=1200)
    # 插入轴
    axins = inset_axes(ax, width="30%", height="30%", loc='lower left',bbox_to_anchor=(0.1, 0.4, 1, 1.1), bbox_transform=ax.transAxes)

    axins.plot(baseline_data[0]['data'][0, :], model_data[1]['data'][4], linewidth=2, marker='o', markersize=5,
               color='r',
               label="Attention-RL")
    print("PointerNet:", model_data[1]['data'][4])

    axins.plot(baseline_data[0]['data'][0, :], model_data[0]['data'][4], linewidth=2, marker='+', markersize=7,
               color='y',
               label="Attention-DARL")
    print("Ours:", model_data[1]['data'][4])
    axins.plot(baseline_data[0]['data'][0, :], model_data[2]['data'][4], linewidth=2, marker='o', markersize=5, color='b',
            label="Pointer-RL")
    print("Ours:", model_data[0]['data'][4])
    axins.set_xticks(baseline_data[0]['data'][0, :])
    axins.set_xticklabels((baseline_data[0]['data'][0, :] * 3).astype(int))
    axins.set_ylim(0, 12)
    axins.set_yticks(range(1, 12))
    axins.grid()
    axins.legend(fontsize=8)

    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    # Partial TIME
    colors = mcolors.TABLEAU_COLORS
    color_name = list(mcolors.TABLEAU_COLORS)
    print("---------- Partial TIME--------------")

    ax.plot(baseline_data[0]['data'][0, :], model_data[1]['data'][4], linewidth=2, marker='o', markersize=5, color='r',
            label="Transformer-RL")
    print("PointerNet:", model_data[1]['data'][4])

    ax.plot(baseline_data[0]['data'][0, :], model_data[0]['data'][4], linewidth=2, marker='+', markersize=7, color='y',
            label="TransDA-RL")
    print("Ours:", model_data[1]['data'][4])

    # ax.set_ylabel('Time [s]', fontsize=15, fontweight='bold')
    # ax.set_xlabel('# of missions', fontsize=15, fontweight='bold')
    ax.set_xticks(baseline_data[0]['data'][0, :])
    ax.set_xticklabels((baseline_data[0]['data'][0, :] * 3).astype(int))
    # ax.legend()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=15)

    ax.grid()
    # fig.savefig('./Archive/Partial_Time_Analyze.png', dpi=1200)

    plt.show()

    if verbose:
        for d in baseline_data:
            print("TYPE:", d['solver_type'])
            print("COST:", d['data'][3, :])
            print("TIME", d['data'][4, :])

    if save:
        plt.savefig('./Archive/Base_VS_Model.png', dpi=300)


def cost_gap_with_baseline(verbose=False, save=False):
    File_paths = []
    type2 = "Data/Balanced_type2_10000_777.pkl"
    File_paths.append(type2)
    type1 = "Data/Balanced_type1_10000_777.pkl"
    File_paths.append(type1)
    greedy = "Data/Balanced_greedy_10000_777.pkl"
    File_paths.append(greedy)

    baseline_data = []
    for file_path in File_paths:
        with open(file_path, 'rb') as f:
            baseline_data.append(pickle.load(f))

    model_data_paths = []
    scale_data = "Data/General_Final_10000_777.pkl"
    model_data_paths.append(scale_data)
    scale_data_Ptr = "Data/Scale_Ptr_10000_777.pkl"
    model_data_paths.append(scale_data_Ptr)

    model_data = []
    for model_data_path in model_data_paths:
        with open(model_data_path, 'rb') as f:
            model_data.append(pickle.load(f))

    colors = mcolors.TABLEAU_COLORS
    color_name = list(mcolors.TABLEAU_COLORS)

    fig, ax = plt.subplots(figsize=(12, 8))

    base_line_cost = baseline_data[0]['data'][3, :]

    print("OR-Type1:", (baseline_data[1]['data'][3, :] - base_line_cost) / base_line_cost * 100)
    print("Greedy:", (baseline_data[2]['data'][3, :] - base_line_cost) / base_line_cost * 100)
    # COST
    # for d in baseline_data[1:]:
    #   ax.plot(d['data'][0,:],(d['data'][3,:]-base_line_cost)/base_line_cost*100,linewidth=1 ,linestyle='--' ,marker='o',markersize=3 ,label=d['solver_type'])
    ax.plot(baseline_data[1]['data'][0, :], (baseline_data[1]['data'][3, :] - base_line_cost) / base_line_cost * 100,
            linewidth=2, color=colors[color_name[0]], linestyle="-.", marker='x', markersize=7, label="OR-Type1")
    ax.plot(baseline_data[2]['data'][0, :], (baseline_data[2]['data'][3, :] - base_line_cost) / base_line_cost * 100,
            linewidth=2, color=colors[color_name[2]], linestyle="-.", marker='s', markersize=4, label="Greedy")

    pn_data = (model_data[1]['cost'] - base_line_cost) / base_line_cost * 100

    ax.plot(baseline_data[0]['data'][0, :], pn_data, linewidth=2, marker='d', markersize=7, color='b',
            label="PointerNet-RL")
    print("PointerNet:", (model_data[1]['cost'] - base_line_cost) / base_line_cost * 100)

    ours_data = (model_data[0]['cost'] - base_line_cost) / base_line_cost * 100

    ax.plot(baseline_data[0]['data'][0, :], ours_data, linewidth=2, marker='o', markersize=5, color='r',
            label="Transformer-RL")
    print("OURS:", (model_data[0]['cost'] - base_line_cost) / base_line_cost * 100)

    ax.set_ylabel('Gap [%]', fontsize=15, fontweight='bold')
    ax.set_xlabel('# of missions', fontsize=15, fontweight='bold')
    ax.set_xticks(baseline_data[0]['data'][0, :])
    ax.set_xticklabels((baseline_data[0]['data'][0, :] * 3).astype(int))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=15)
    # ax.legend(fontsize=15)
    ax.grid()

    # if verbose:
    #   for d in baseline_data[1:]:
    #     ax.plot(d['data'][0,:],(d['data'][3,:]-base_line_cost)/base_line_cost*100,linewidth=1 ,linestyle='--' ,marker='o',markersize=3 ,label=d['solver_type'])
    plt.show()
    fig.savefig('./Gap_Analyze.png', dpi=1200)

    # Partial gap
    fig, ax = plt.subplots(figsize=(8, 5))

    base_line_cost = baseline_data[0]['data'][3, :]

    pn_data = (model_data[1]['cost'] - base_line_cost) / base_line_cost * 100
    # pn_data[-1] = 3.5

    ax.plot(baseline_data[0]['data'][0, :], pn_data, linewidth=2, marker='d', markersize=7, color='b',
            label="PointerNet-RL")

    print("PointerNet:", (model_data[1]['cost'] - base_line_cost) / base_line_cost * 100)

    ours_data = (model_data[0]['cost'] - base_line_cost) / base_line_cost * 100
    # ours_data[1] = 2.0
    # ours_data[3] = 2.3

    ax.plot(baseline_data[0]['data'][0, :], ours_data, linewidth=2, marker='o', markersize=5, color='r',
            label="Transformer-RL")
    print("OURS:", (model_data[0]['cost'] - base_line_cost) / base_line_cost * 100)

    ax.set_ylabel('Gap [%]', fontsize=15, fontweight='bold')
    ax.set_xlabel('# of missions', fontsize=15, fontweight='bold')
    ax.set_xticks(baseline_data[0]['data'][0, :])
    ax.set_xticklabels((baseline_data[0]['data'][0, :] * 3).astype(int))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=15)
    # ax.legend(fontsize=15)
    ax.grid()

    # if verbose:
    #   for d in baseline_data[1:]:
    #     ax.plot(d['data'][0,:],(d['data'][3,:]-base_line_cost)/base_line_cost*100,linewidth=1 ,linestyle='--' ,marker='o',markersize=3 ,label=d['solver_type'])
    plt.show()
    # fig.savefig('./Archive/Partial_Gap_Analyze.png', dpi=1200)


def cost_generalization(verbose=False, save=False):
    with open("Data/Balanced_type2_10000_777.pkl", 'rb') as f:
        baseline_data = pickle.load(f)

    with open("Data/Beta_Gneral_10000_777.pkl", 'rb') as f:
        model_data = pickle.load(f)

    ax = plt.subplot()

    base_line_cost = baseline_data['data'][3, :]
    model_cost = model_data['cost']

    total_gap = (model_cost - base_line_cost) / base_line_cost * 100

    # COST
    for i in range(len(model_cost)):
        ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], total_gap[i, :], marker='o',
                label='Trained with %d tasks' % ((i + 1) * 3), linewidth=1, markersize=2)

    ax.set_ylabel('%', fontsize=15, fontweight='bold')
    ax.set_xlabel('Task number', fontsize=15, fontweight='bold')
    ax.set_xticks(baseline_data['data'][0, :])
    ax.set_xticklabels((baseline_data['data'][0, :] * 3).astype(int))
    ax.set_title('Cost gap(VS Type2)')
    ax.legend(fontsize=15)
    ax.grid()

    plt.show()


def reward_incline(verbose=False, save=False):
    reward_data = 'rewards_epoch/HetNet'


if __name__ == "__main__":
    cost_baseline_and_model_performance(verbose=True, save=False)
    # cost_gap_with_baseline(verbose=True, save=False)
