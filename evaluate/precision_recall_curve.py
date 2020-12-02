import numpy as np
from argparse import ArgumentParser
import math
import json
import matplotlib.pyplot as plt

def calculate_PR(degrees_dictionary1, degrees_dictionary2, threshold):
    """
    threshold: degrees
    """
    keys1 = list(degrees_dictionary1.keys())
    keys1.sort(key=lambda x: int(x))
    keys2 = list(degrees_dictionary2.keys())
    keys2.sort(key=lambda x: int(x))

    TP = np.sum([degrees_dictionary1[key] for key in keys1 if int(key) <= threshold])
    FN = np.sum([degrees_dictionary1[key] for key in keys1 if int(key) > threshold])

    FP = np.sum([degrees_dictionary2[key] for key in keys2 if int(key) < threshold])
    TN = np.sum([degrees_dictionary2[key] for key in keys2 if int(key) >= threshold])

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)

    return precision, recall, accuracy


if __name__ == '__main__':
    parser = ArgumentParser(description="calculate_accuray")
    parser.add_argument("-positive_pairs", "--positive_pairs",
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/result/positive_pairs.json",
                        help="specify your json file that contain results of positive pairs", type=str)
    parser.add_argument("-negative_pairs", "--negative_pairs",
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/result/negative_pairs.json",
                        help="specify your json file that contain results of negative pairs", type=str)
    parser. add_argument('-figure', '--figure', help='specify file path to save precision-recall curve figure',
                         default="/home/phamvanhanh/PycharmProjects/FaceVerification/result/precision_recall_curve.png", type=str)
    args = parser.parse_args()
    degrees_dictionary1 = json.loads(open(args.positive_pairs, 'r').read())
    degrees_dictionary2 = json.loads(open(args.negative_pairs, 'r').read())

    # precision, recall = calculate_PR(degrees_dictionary1, degrees_dictionary2, 70)

    thresholds = np.arange(10, 100, 5)[::-1]
    cosins = np.cos(thresholds)

    precisions = list()
    recalls = list()
    for threshold in thresholds:
        pre, re, _ = calculate_PR(degrees_dictionary1, degrees_dictionary2, threshold)
        precisions.append(pre)
        recalls.append(re)

    f1_scores = 2 / ((1/np.array(precisions)) + (1/np.array(recalls)))
    args_max = np.argmax(f1_scores)
    chose_threshold = thresholds[args_max]
    _, __, accuracy = calculate_PR(degrees_dictionary1, degrees_dictionary2, chose_threshold)
    print("Accuracy: {:.2f}".format(accuracy))
    print("F1-Score: {:.2f}".format(f1_scores[args_max]))
    print("Chose_threshold: ", chose_threshold)



    plt.plot(recalls, precisions, color='darkorange', lw=2)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(args.figure)







