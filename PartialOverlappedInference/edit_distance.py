import numpy as np
import numba
from typing import List, Tuple

W_INS = 2
W_INS_NON = 0
W_DEL = 2
W_DEL_NON = 0
W_SUB = 1
W_MATCH = -2


def alloc(Di, Di_1):
    C = np.zeros([len(Di) + 1, len(Di_1) + 1], dtype=np.int32)
    Di = np.asarray(Di)
    Di_1 = np.asarray(Di_1)
    return C, Di, Di_1


@numba.jit(nopython=True)
def row_fill(C):
    for j in range(0, C.shape[0]):  # 0 to N_i + 1
        C[j, 0] = W_DEL_NON * j


@numba.jit(nopython=True)
def col_fill(C):
    for k in range(1, C.shape[1]):  # 1 to N_{i+1} + 1
        C[0, k] = W_INS * k


@numba.jit(nopython=True)
def e_sub(Di, Di_1, j, k):
    # j/k indices are padded by 1, so subtract 1 when indexing seq Di and Di_1
    if Di[j - 1] == Di_1[k - 1]:
        return W_MATCH
    else:
        return W_SUB


@numba.jit(nopython=True)
def cost(C, Di, Di_1):
    # first pass
    for j in range(1, C.shape[0]):  # 1 to N_i + 1
        for k in range(1, C.shape[1]):  # 1 to N_{i+1} + 1
            if j < (C.shape[0] - 1):
                del_cost = C[j - 1, k] + W_DEL
                ins_cost = C[j, k - 1] + W_INS
                sub_cost = C[j - 1, k - 1] + e_sub(Di, Di_1, j, k)

                C[j, k] = min(del_cost, ins_cost, sub_cost)
            else:
                del_cost = C[j - 1, k] + W_DEL
                ins_cost = C[j, k - 1] + W_INS_NON
                sub_cost = C[j - 1, k - 1] + e_sub(Di, Di_1, j, k)

                C[j, k] = min(del_cost, ins_cost, sub_cost)


def compute_alignment(Di: str, Di_1: str):
    Di = list(Di)
    Di_1 = list(Di_1)

    # allocate memory
    C, Di, Di_1 = alloc(Di, Di_1)

    # initialize
    row_fill(C)
    col_fill(C)

    # compute cost
    cost(C, Di, Di_1)
    return C


@numba.jit(nopython=True)
def compute_overlap_path(C) -> (List[Tuple[int]], Tuple[int]):
    j, k = C.shape[0] - 1, C.shape[1] - 1
    idx = None
    path = [(j, k)]

    while j > 0 or k > 0:
        if j > 0 and k > 0:
            top = C[j - 1, k]
            left = C[j, k - 1]
            diagonal = C[j - 1, k - 1]

            if diagonal <= top and diagonal <= left:
                # overlapped segment, update both idx and path
                idx = (j - 1, k - 1)
                path.append((j - 1, k - 1))
                j = j - 1
                k = k - 1

            elif top <= left and top <= diagonal:
                # dont update overlap index, just path
                path.append((j - 1, k))
                j = j - 1

            elif left <= top and left <= diagonal:
                # not overlap, but prioritize Di_1 so update idx
                idx = (j, k - 1)
                path.append((j, k - 1))
                k = k - 1

            else:
                print("[INVALID STATE DURING ALIGNMENT BACKTRACK; EXITING]")
                break

        elif j > 0:
            # dont update overlap index, just path
            path.append((j - 1, k))
            j = j - 1

        else:
            # not overlap, but prioritize Di_1 so update path but not idx
            path.append((j, k - 1))
            k = k - 1

    return path, idx


def merge_text(Di, Di_1, overlap_idx):
    new_seq = Di[:overlap_idx[0]] + Di_1[overlap_idx[1]:]
    return new_seq


def print_alignment(C):
    for j in range(C.shape[0]):  # 0 to N_i + 1
        for k in range(C.shape[1]):  # 0 to N_{i+1} + 1
            print(f"{C[j][k]}\t", end="")
        print()
    print()


if __name__ == '__main__':
    Di = "speech recognize"
    Di_1 = "cognition"
    print("Previous Buffer (Di) :", Di)
    print("New Buffer    (Di+1) :", Di_1)

    C = compute_alignment(Di, Di_1)

    print()
    print("Alignment Matrix :")
    print_alignment(C)

    path, overlap_idx = compute_overlap_path(C)
    path = [str(p) for p in path]

    print("Overlap path : ", " -> ".join(path))
    print("Overlap index :", overlap_idx)

    new_sentence = "".join(merge_text(Di, Di_1, overlap_idx))
    print()
    print("Previous Buffer (Di) :", Di)
    print("New Buffer    (Di+1) :", Di_1)
    print("Merged sequence      :", new_sentence)
