import pprint

def is_tree(heads):
    "Verify if the directed graph codified in a list, where the position represents the dependente and the value represents the head, is a tree"  
    h = [-1]
    n = len(heads)
    for i in range(1, n):
        if heads[i] < 0 or heads[i] > n:
            return False
        h.append(-1)
    for i in range(1, n):
        k = i
        while k > 0:
            if h[k] >= 0 and h[k] < i :
                break
            if h[k] == i:
                return False
            h[k] = i
            k = heads[k]
    return True

def transform(heads, i, j):
    h = heads.copy()
    h[j] = i
    return h


def test_transform(y = [-1, 2, 0, 4, 2, 2, 7, 5]):
    n = len(y)
    print("is {} a tree? {}".format(y, is_tree(y)))
    pprint.pprint([ [ is_tree(transform(y, i, j)) for i in range(0, n)] for j in range(1,n)])
