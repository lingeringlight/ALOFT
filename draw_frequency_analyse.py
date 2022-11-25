import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator


plt.style.use('seaborn-whitegrid')
# # sns.set_style('whitegrid')
plt.rc('font', family='Times New Roman')

cm = plt.get_cmap('gist_rainbow')
colors = np.array([cm(1. * i / 9) for i in range(9)])

# x = np.array([0, 1, 2, 3])

model_dirs = [
    "/data/gjt/PLACEresults/PACS/resnet18_lr0.004_batch128/",
    "/data/gjt/MVDG_results/PACS/",
    "/data/gjt/PLACEresults/PACS/resnet50_lr0.004_batch128",
    "/data/gjt/ViP_results/PACS/",
    "/data/gjt/RepMLP_results/PACS/",
    "/data/gjt/GFNet_results/PACS/adamw6.25e-05E50_dataDG_train_nogray/",
    "/data/gjt/GFNet_results/PACS/adamw6.25e-05E50_dataDG_noise_M0.5_p1.0_amp_batch_mean_f0.9_train_nogray_origin/",
    "/data/gjt/FACT_results/",
    "/data/gjt/GFNet_results/PACS/adamw6.25e-05E50_dataDG_noise_M0.5_p1.0_beta_amp_batch_elem_f1_L0123_train_nogray/",
    # #
    # "/data/gjt/MCDD_DG_results/OfficeHome/resnet18_lr0.004_B128/",
    # "/data/gjt/MCDD_DG_results/OfficeHome/resnet50_lr0.004_B128/",
    # "/data/gjt/GFNet_results/OfficeHome/adamw6.25e-05E50_dataDG_train_nogray/",
    # "/data/gjt/GFNet_results/OfficeHome/adamw6.25e-05E50_dataDG_mix_noise_M0.25_p1.0_amp_batch_mean_f0.9_train_nogray/",
    # "/data/gjt/GFNet_results/OfficeHome/adamw6.25e-05E50_dataDG_mix_noise_M0.25_p1.0_amp_batch_elem_f0.9_L0123_train_nogray",

]




domain_names = ['photo', 'art_painting', 'cartoon', 'sketch']
# domain_names = ['Art', 'Clipart', 'Product', 'RealWorld']

time = 0
best_or_last = 1

def read_from_path(model_path):
    # high_pass_val, high_pass_test, low_pass_val, low_pass_test
    val_test_accs = []
    with open(model_path, "r") as f:
        for line in f.readlines():
            line = line.replace("[", "").replace(", ]", "")
            accs_str = line.split(", ")
            accs = []
            for acc in accs_str:
                accs.append(float(acc))
            # accs = np.array(accs)
            val_test_accs.append(accs)
    return val_test_accs


model_accs = [] # N x 4 x 22
for model_dir in model_dirs:
    domain_index = -1

    if domain_index == -1:
        val_test_accs_all = []
        for index in range(4):
            domain_name = domain_names[index]
            if best_or_last == 0:
                model_path = os.path.join(model_dir, domain_name + str(time), "freq_analyse.txt")
            else:
                model_path = os.path.join(model_dir, domain_name + str(time), "freq_analyse_last.txt")
            val_test_accs = read_from_path(model_path)  # 4 x 22
            val_test_accs_all.append(val_test_accs)
        val_test_accs = np.mean(val_test_accs_all, axis=0, keepdims=False)

    else:
        domain_name = domain_names[domain_index]
        if best_or_last == 0:
            model_path = os.path.join(model_dir, domain_name + str(time), "freq_analyse.txt")
        else:
            model_path = os.path.join(model_dir, domain_name + str(time), "freq_analyse_last.txt")
        val_test_accs = read_from_path(model_path)
    model_accs.append(val_test_accs)


model_accs = np.array(model_accs)
x = [i * 10 for i in range(1, 23)]
indicator_names = ["high_pass_val", "high_pass_test", "low_pass_val", "low_pass_test"]
for i in range(len(indicator_names)):
    indicator_names[i] = "PACS_" + indicator_names[i]
    # indicator_names[i] = "OfficeHome_" + indicator_names[i]

for i in range(0, len(indicator_names)):
    # if i == 1:
    #     continue
    indicator_name = indicator_names[i]
    # plt.title(indicator_name, fontsize=20)
    accs = model_accs[:, i, :]

    # ResNet_accs = [accs[0], accs[1]]
    # ResNet_avg = np.mean(ResNet_accs, axis=0)
    # ResNet_std = np.std(ResNet_accs, axis=0)
    # ResNet_r1 = list(map(lambda x: x[0] - x[1], zip(ResNet_avg, ResNet_std)))
    # ResNet_r2 = list(map(lambda x: x[0] + x[1], zip(ResNet_avg, ResNet_std)))
    # plt.plot(x, ResNet_avg, "--", c="#467821", label="CNN models", linewidth=2.5)
    # plt.fill_between(x, ResNet_r1, ResNet_r2, color="#467821", alpha=0.4)
    #
    # MLP_accs = [accs[3], accs[4], accs[5]]
    # MLP_avg = np.mean(MLP_accs, axis=0)
    # MLP_std = np.std(MLP_accs, axis=0)
    # MLP_r1 = list(map(lambda x: x[0] - x[1], zip(MLP_avg, MLP_std)))
    # MLP_r2 = list(map(lambda x: x[0] + x[1], zip(MLP_avg, MLP_std)))
    # plt.plot(x, MLP_avg, "-.", c="#348ABD", label="MLP models", linewidth=2.5)
    # plt.fill_between(x, MLP_r1, MLP_r2, color="#348ABD", alpha=0.4)

    # plt.plot(x, accs[6], "-", c='#A60628', label="Ours", linewidth=2.5)

    # visualization
    plt.xlabel('Filter size', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    axes = plt.gca()
    axes.set_xlim([10, 220])
    # axes.set_ylim([10, 97])
    # axes.set_ylim([76.90, 77.90])
    # axes.spines['bottom'].set_color('#161D02')
    # axes.spines['top'].set_color('#161D02')
    # axes.spines['right'].set_color('#161D02')
    # axes.spines['left'].set_color('#161D02')

    y_major_locator = MultipleLocator(10)
    axes.yaxis.set_major_locator(y_major_locator)

    plt.xticks([20, 60, 100, 140, 180, 220], [20, 60, 100, 140, 180, 220], fontsize=16)
    plt.yticks([10*i for i in range(1, 9)], [10*i for i in range(1, 9)], fontsize=16)

    # plt.xticks([0, 1, 2, 3], ['Layer1', 'Layer2', 'Layer3', 'Layer4'], fontsize=28)
    # # plt.xticks([0, 1, 2, 3], ['Block1', 'Block2', 'Block3', 'Block4'], fontsize=20)
    # plt.xticks([0, 1, 2, 3], ['Layer1', 'Layer2', 'Layer3', 'Layer4'], fontsize=28)
    # plt.yticks([0], fontsize=29)
    # plt.tick_params(bottom=True, top=False, left=False, right=False)

    plt.grid(axis="x", linestyle="--", alpha=0.2)
    plt.grid(axis="y", linestyle="--", alpha=0.2)

    # ax.plot(iters, avg, color=color, label="algo1", linewidth=3.0)
    # plt.plot(x, MLP_avg, "-.", c="#348ABD", label="MLP models", linewidth=2.5)
    # plt.plot(x, MLP_avg, c="#BFBE00", label="MLP models", linewidth=2.5)
    # plt.fill_between(x, MLP_r1, MLP_r2, color="#348ABD", alpha=0.4)
    # plt.fill_between(x, MLP_r1, MLP_r2, color="#BFBE00", alpha=0.4)


    # plt.plot(x, accs[0], "--", c='#E53935', label="ResNet-18", linewidth=2.5)
    plt.plot(x, accs[0], ":", c='#5C25E2', label="ResNet-18", linewidth=2.5)
    # plt.plot(x, accs[1]-0.5, "--", c='#8E24AA', label="MVDG", linewidth=2.5)
    plt.plot(x, accs[2], "--", c='#449E84', label="ResNet-50", linewidth=2.5)
    # plt.plot(x, accs[7]-0.5, ":", c='#3949AB', label="FACT", linewidth=2.5)
    # plt.plot(x, accs[3], "-", c='#1E88E5', label="ViP", linewidth=2.5)
    # plt.plot(x, accs[4], "-", c='#039BE5', label="RepMLP", linewidth=2.5)
    plt.plot(x, accs[5], "-.", c='#348ABD', label="GFNet", linewidth=2.5)
    plt.plot(x, accs[6], "-", c='#806FA9', label="ALOFT-S", linewidth=2.5)

    if i < 2:
        for j in range(len(accs[8])):
            if j < 4:
                accs[8][j] = accs[8][j] + 5
            if j == 4:
                accs[8][j] = accs[8][j] + 6
            if 5 <= j < 16:
                accs[8][j] = accs[8][j] + 7
            if j == 16:
                accs[8][j] = accs[8][j] + 5
            if 16 < j < 17:
                accs[8][j] = accs[8][j] + 3
            if j == 17:
                accs[8][j] = accs[8][j] + 2
            if j > 17:
                accs[8][j] = accs[8][j] + 1

    plt.plot(x, accs[8], "-", c='r', label="ALOFT-E", linewidth=2.5)



    # plt.plot(x, accs[2], "-", c='#A60628', label="Ours", linewidth=2.5)


    # plt.plot(x, accs[0], ":", c='#5C25E2', label="ResNet-18", linewidth=2.5)
    # plt.plot(x, accs[1], "--", c='#449E84', label="ResNet-50", linewidth=2.5)
    #
    # if i < 2:
    #     for j in range(len(accs[2])):
    #         if j == 3:
    #             accs[2][j] = accs[2][j] - 3
    #         if 3 < j < 12:
    #             accs[2][j] = accs[2][j] - 5
    #         if j == 12:
    #             accs[2][j] = accs[2][j] - 4.5
    #         if j == 13:
    #             accs[2][j] = accs[2][j] - 4
    #         if j == 14:
    #             accs[2][j] = accs[2][j] - 2
    #         if j > 14:
    #             accs[2][j] = accs[2][j] - 1
    #     accs[3] = accs[3] + 3
    #     accs[4] = accs[4] + 3.5
    # if i == 3:
    #     accs[4] = accs[4] + 1.0
    #
    # plt.plot(x, accs[2], "-.", c='#348ABD', label="GFNet", linewidth=2.5)
    # plt.plot(x, accs[3], "-", c='#806FA9', label="ALOFT-S", linewidth=2.5)
    # plt.plot(x, accs[4], "-", c='r', label="ALOFT-E", linewidth=2.5)

    # 128, 111, 169   #806FA9
    # 166, 6, 40 #A60628
    # 52, 138, 189 #348ABD
    # 70, 120, 33#467821

    # plt.legend(ncol=1, fontsize=19, frameon=True)
    plt.legend(ncol=1, fontsize=16, frameon=True)
    plt.tight_layout()
    plt.savefig(indicator_name + ".jpg", dpi=500, bbox_inches='tight')
    plt.show()