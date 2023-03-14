import itertools
import random

def print_list_permutations(perm_list, amount_perm):
    curr_amount = 1
    for permutation in itertools.permutations(perm_list):
        print(f"Permutation {curr_amount}:")
        print(list(permutation))
        curr_amount += 1
        if curr_amount >= amount_perm:
            break


def print_list_permutations_simple(perm_list, amount_perm):

    out_list = []
    for i in range(1, amount_perm + 1):


        print(f"Permutation {i}:")
        shuffled = list(perm_list)
        random.shuffle(shuffled)
        while(shuffled in out_list):
            random.shuffle(shuffled)

        out_list.append(shuffled)
        print(shuffled)



if __name__ == "__main__":
    # print_list_permutations([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33], 10)

    print_list_permutations_simple([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33], 10)