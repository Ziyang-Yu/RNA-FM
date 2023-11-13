import torch
import os
import numpy as np
import collections
import _pickle as cPickle




def get_prob_from_results(folder_path: str):

    res_dict = dict()

    for filename in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Check if it is a file (not a subfolder)
        if os.path.isfile(file_path):
            rna_name = filename.replace(".npy", "")
            data = np.load(file_path)
            data = torch.from_numpy(data)
            res_dict[rna_name] = data
    return res_dict


def get_pair_from_npy(file_paths: list):

    res_dict = dict()

    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            loaded_data = cPickle.load(file)
            for data in loaded_data:
                res_dict[data.name.replace(".ct", "")] = data
    return res_dict

def get_pair_using_top5(prob, label):
    prob = prob.cuda()
    seqpos = torch.arange(label.length, device='cuda')
    x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
    valid_mask = ((y_ind - x_ind) >= 6).cuda()
    masked_prob = (prob * valid_mask).view(-1)
    most_likely = masked_prob.topk(label.length // 5, sorted=False)

    pairs = label.pairs
    label = torch.zeros((label.length, label.length)).cuda()

    for pair in pairs:
        label[pair[0], pair[1]] = 1
        label[pair[1], pair[0]] = 1
    selected = label.view(-1).gather(0, most_likely.indices)
    correct = selected.sum().float().cpu().detach().numpy()


    total = selected.numel()
    return correct, total


if __name__ == "__main__":

    pred_path = "/home/eric/RNA-FM/recover_experiments/e2efold_data/results/r-ss"
    labels_path = ["/home/eric/RNA-FM/recover_experiments/e2efold_data/archiveII_all/all_600.pickle", 
                    "/home/eric/RNA-FM/recover_experiments/e2efold_data/archiveII_all/all_1800.pickle"]

    RNA_SS_data = collections.namedtuple('RNA_SS_data', 
        'seq ss_label length name pairs')


    probs = get_prob_from_results(pred_path)
    labels = get_pair_from_npy(labels_path)

    correct = 0
    total = 0
    for name in list(probs.keys()):
        prob = probs[name]
        label = labels[name]
        correct_temp, total_temp = get_pair_using_top5(prob, label)
        correct += correct_temp
        total += total_temp
        # print(correct, total)
        # break
        

    print(correct/total)