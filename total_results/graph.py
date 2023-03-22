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


ver = [512]
"""
version
"""
# train_type = "original"
# train_type = "norm_first"
train_type = "linear"

for version in ver:
    version = str((version, train_type))
    print("The version is " + version)

    def draw(mode, version):
        version = str(version)
        if mode == "loss":
            train = read("results/train_loss-" + version + ".txt")
            test = read("results/test_loss-" + version + ".txt")
            plt.plot(train, "r", label="train")
            plt.plot(test, "b", label="validation")
            plt.legend(loc="lower left")

        elif mode == "bleu":
            bleu = read("results/bleu-" + version + ".txt")
            plt.plot(bleu, "b", label="bleu score")
            plt.legend(loc="lower right")

        plt.xlabel("epoch")
        plt.ylabel(mode)
        plt.title("training result")
        plt.grid(True, which="both", axis="both")
        plt.savefig("graphs/" + mode + "-" + version)
        # plt.show()
        plt.cla()

    if __name__ == "__main__":
        draw(mode="loss", version=version)
        draw(mode="bleu", version=version)
