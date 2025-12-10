from plotter import plot_result
from Environment import Mission
from Model_Loader import load_model_eval
from torch.utils.data import DataLoader
import numpy as np
import argparse

case = 5
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="hetnet")
parser.add_argument("--coverage_num", type=int, default=case)
parser.add_argument("--visiting_num", type=int, default=case)
parser.add_argument("--pick_place_num", type=int, default=case)
args = parser.parse_args()


X = Mission.MissionDataset(num_visit=case, num_coverage=case, num_pick_place=case,
                           num_samples=1,
                           random_seed=100,
                           overlap=True)

data_loader = DataLoader(X, batch_size=1, shuffle=True)

model_type = 'HetNet' + '_C' + str(case) + '_V' + str(case) + '_D' + str(case)

model = load_model_eval('Models/O_HetNet/' + model_type + '.param',
                        'Models/O_HetNet/' + model_type + '.config')

for (indices, sample_batch) in data_loader:
    r, p, a = model(sample_batch)
    plot_result(np.array(X.data_set[0]), np.array(a).reshape(-1), args)
