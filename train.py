from csp_utils import CSP_Instance, mc_weighted_language
from model import RUN_CSP

from utils import GraphDataset,mk_dir
import argparse
import numpy as np
import networkx as nx
import numpy as np
from tqdm import tqdm


def train(network, train_data, t_max, epochs):
    """
    Trains a RUN-CSP Network on the given data
    :param network: The RUN_CSP network
    :param train_data: A list of CSP instances that are used for training
    :param t_max: Number of RUN_CSP iterations on each instance
    :param epochs: Number of training epochs
    """

    best_conflict_ratio = 1.0
    for e in range(epochs):
        print('Epoch: {}'.format(e))

        # train one epoch
        output_dict = network.train(train_data, iterations=t_max)
        conflict_ratio = output_dict['conflict_ratio']
        print(f'Ratio of violated constraints: {conflict_ratio}')

        # if network improved, save new model
        if conflict_ratio < best_conflict_ratio:
            network.save_checkpoint('best')
            best_conflict_ratio = conflict_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', type=str,  help='Distribution of dataset')
    parser.add_argument('-t', '--t_max', type=int, default=30, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-s', '--state_size', type=int, default=128, help='Size of the variable states in RUN-CSP')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size used during training')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of training epochs')

    args = parser.parse_args()
    language = mc_weighted_language
    train_graph_gen=GraphDataset(folder_path=f'../data/training/{args.distribution}')
    print(f'Number of graphs:{len(train_graph_gen)}')
    graphs = [nx.from_numpy_array(train_graph_gen.get()) for _ in range(len(train_graph_gen))]
    
    train_instances = [CSP_Instance.graph_to_weighted_mc_instance(g) for g in tqdm(graphs)]
    del graphs
    train_batches = CSP_Instance.batch_instances(train_instances,32)
    del train_instances
    
    model_save_path=f'models/{args.distribution}'
    mk_dir(model_save_path)
    # create RUN_CSP instance for given constraint language
    network = RUN_CSP(model_save_path, language)

 

    # train and store the network
    train(network, train_batches, args.t_max, args.epochs)


if __name__ == '__main__':
    main()
