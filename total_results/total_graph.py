"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, "r")
    file = f.read()
    file = re.sub("\\[", "", file)
    file = re.sub("\\]", "", file)
    f.close()
    return [float(i) for idx, i in enumerate(file.split(","))]


versions = [512]
types = [
    # "original",
    # "norm_first",
    "linear",
    "gradient_only_RES",
]


def draw(mode):
    for ver in versions:
        for train_type in types:
            version = str((ver, train_type))
            # print("++++++++++++++++++++++++")
            # print(train_type)
            # print(version)
            if mode == "loss":
                train = read("results/train_loss-" + version + ".txt")
                test = read("results/test_loss-" + version + ".txt")
                plt.plot(train, "r", label="train")
                plt.plot(test, "b", label="validation")
                plt.legend(loc="lower left")

            elif mode == "bleu":
                bleu = read("results/bleu-" + version + ".txt")
                plt.plot(bleu, label="bleu score" + version)
                plt.legend(loc="lower right")
                # plt.xlim([0, 10])  # X축의 범위: [xmin, xmax]
                # plt.ylim([0, 20])  # Y축의 범위: [ymin, ymax]

    plt.xlabel("epoch")
    plt.ylabel(mode)
    plt.title("training result")
    plt.grid(True, which="both", axis="both")
    plt.savefig("graphs/total_" + mode + "-" + version)
    plt.show()


if __name__ == "__main__":
    # draw(mode="loss", version=version)
    draw(mode="bleu")
