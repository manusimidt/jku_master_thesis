import argparse
import copy
import time
from multiprocessing import Pool


def main(args, i):
    print(args)
    time.sleep(20)
    return f"FINISHED FOR SEED {args['seed']}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", choices=["vanilla", "random", "UCB"], default="vanilla",
                        help="The environment to train on")
    parser.add_argument("-c", "--conf", choices=["narrow_grid", "wide_grid"], default="wide_grid",
                        help="The environment configuration to train on")
    parser.add_argument("-a", "--alg", choices=["PPO", "DRAC"], default="PPO",
                        help="The algorithm to train with")
    parser.add_argument("-s", "--seed", default=[1, 2, 3], type=int, help="seed", nargs='+')

    args = parser.parse_args()
    hyperparams = vars(args)

    if len(args.seed) == 1:
        main(hyperparams, 4)
    else:
        hyperparams_arr = []
        for i in range(len(args.seed)):
            hyperparams_arr.append(copy.deepcopy(hyperparams))
            hyperparams_arr[-1]['seed'] = args.seed[i]
        with Pool(5) as p:
            results = p.map(main, hyperparams_arr)
            print(results)
