__author__ = 'Daan van Stigt'

import multiprocessing as mp

import numpy as np

import dependencytree

from perceptron import ArcPerceptron, LabPerceptron
from decode import Decoder
from features import get_features
from parallel import parse_parallel
from utils import softmax



class DependencyParser:
    def __init__(self, feature_opts={}, decoding='mst'):
        self.feature_opts = feature_opts
        self.arc_perceptron = ArcPerceptron(self.feature_opts)
        self.decoder = Decoder(decoding)
        self.arc_accuracy = None

    def make_features(self, lines):
        self.arc_perceptron.make_features(lines)

    def parse(self, tokens):
        # TODO Fix weird UD parsing! What to do with this 20.1 id business!?
        score_matrix = np.zeros((len(tokens), len(tokens)))
        all_features = dict()
        for i, dep in enumerate(tokens):
        # for dep in tokens:
            all_features[i] = dict()
            all_features[i] = dict()
            for j, head in enumerate(tokens):
            # for head in tokens:
                features = get_features(head, dep, tokens, **self.feature_opts)
                score = self.arc_perceptron.score(features)
                # score_matrix[dep.id][head.id] = score  # TODO: UD parsing
                # all_features[dep.id][head.id] = features  # TODO: UD parsing
                score_matrix[i][j] = score
                all_features[i][j] = features
        probs = softmax(score_matrix)
        heads = self.decoder(probs)
        #heads = self.first_order_proj_approx(heads, probs.T)
        heads = self.first_order_proj_approx_beam(heads, probs.T)
        return heads, probs, all_features

    def score_heads(self, heads, probs):
        value_proj = 0.0
        for m in range(1, len(heads)):
            h = heads[m]
            value_proj += probs[h, m]
        return value_proj
    
    def first_order_proj_approx(self, heads, probs):
        M = 10
        while True:
            m = float('-inf')
            c = p = -1
            n = len(heads)
            heads_score = self.score_heads(heads, probs)
            for j in range(1, n):
                for i in range(0, n):
                    heads_prime = dependencytree.transform(heads, i, j)
                    if not dependencytree.is_tree(heads_prime):
                        continue
                    delta_score = self.score_heads(heads_prime, probs) - heads_score
                    if delta_score > m:
                        m = delta_score
                        c = j
                        p = i
            if m > 0:
                heads = dependencytree.transform(heads, p, c)
            else:
                return heads
            count_transformations += 1
            if count_transformations >= M:
                return heads

    def get_score(self, item):
        return item[1]

    def next_best_heads(self, heads, probs):
        m = float('-inf')
        c = p = -1
        n = len(heads)
        heads_score = self.score_heads(heads, probs)
        for j in range(1, n):
            for i in range(0, n):
                heads_prime = dependencytree.transform(heads, i, j)
                if not dependencytree.is_tree(heads_prime):
                    continue
                delta_score = self.score_heads(heads_prime, probs) - heads_score
                if delta_score > m:
                    m = delta_score
                    c = j
                    p = i
        if m > 0:
            return dependencytree.transform(heads, p, c)
        else:
            return heads
            
    def first_order_proj_approx_beam(self, heads, probs):
        M = 10
        LIMIT_HEURISTIC = 3
        count_transformations = 0
        heads_score = self.score_heads(heads, probs)
        cid = 1
        count_heuristic = 1
        partial_heads = [[heads, heads_score, cid], [heads, heads_score, cid], [heads, heads_score, cid]]
        n = len(heads)

        heads = partial_heads[0][0]
        for j in range(1, n):
            for i in range(0, n):
                heads_prime = dependencytree.transform(heads, i, j)
                if not dependencytree.is_tree(heads_prime):
                    continue
                score_heads_prime = self.score_heads(heads_prime, probs)
                delta_score = score_heads_prime - partial_heads[-1][1]
                if delta_score > 0:
                    cid += 1
                    partial_heads[-1] = [heads_prime, score_heads_prime, cid]
                    partial_heads = sorted(partial_heads, key=self.get_score)
        set_cids = {partial_heads[0][2], partial_heads[1][2], partial_heads[2][2]}        
        while True:
            set_cids = {partial_heads[0][2], partial_heads[1][2], partial_heads[2][2]}
            heads = partial_heads[0][0]
            for i in range(0, len(partial_heads)):
                heads_prime = self.next_best_heads(partial_heads[i][0], probs)
                score_heads_prime = self.score_heads(heads_prime, probs)
                delta_score = score_heads_prime - partial_heads[i][1]
                if delta_score > 0:
                    cid += 1
                    partial_heads[-1] = [heads_prime, score_heads_prime, cid] #-1
                    partial_heads = sorted(partial_heads, key=self.get_score)
            
            if not set([item[2] for item in partial_heads]).difference(set_cids):
                if count_heuristic >= LIMIT_HEURISTIC :
                    return partial_heads[0][0]
                else:
                    count_heuristic += 1
            else:
                count_heuristic = 1
            count_transformations += 1
            if count_transformations >= M:
                return partial_heads[0][0]

    

    def train(self, niters, train_set, dev_set, approx=100, structured=None):
        # TODO: no structured training yet.
        # Train arc perceptron first.
        for i in range(1, niters+1):
            # Train arc perceptron for one epoch.
            c, n = self.arc_perceptron.train(niters, train_set)
            # Evaluate arc perceptron.
            train_acc, dev_acc = self.evaluate(train_set[:approx]), self.evaluate(dev_set[:approx])
            print(f'| Iter {i} | Correct guess {c:,}/{n:,} ~ {c/n:.2f} | Train UAS {train_acc:.2f} | Dev UAS {dev_acc:.2f} |')
            np.random.shuffle(train_set)

    def train_parallel(self, niters, lines, dev_set, nprocs=-1, approx=100):
        """Asynchronous lock-free (`Hogwild`) training of perceptron."""
        size = mp.cpu_count() if nprocs == -1 else nprocs
        print(f'Hogwild training with {size} processes...')
        # Train arc-perceptron first.
        self.arc_perceptron.prepare_for_parallel()
        for i in range(1, niters+1):
            # Train arc perceptron for one epoch in parallel.
            self.arc_perceptron.train_parallel(niters, lines, size)
            # Evaluate arc perceptron.
            train_acc = self.evaluate_parallel(lines[:approx])
            dev_acc = self.evaluate_parallel(dev_set[:approx])
            print(f'| Iter {i} | Train UAS {train_acc:.2f} | Dev UAS {dev_acc:.2f} |')
            np.random.shuffle(lines)
        self.arc_perceptron.restore_from_parallel()
        # Train lab-perceptron second.

    def accuracy(self, pred, gold):
        return 100 * sum((p == g for p, g in zip(pred, gold))) / len(pred)

    def evaluate(self, lines):
        acc = 0.0
        for line in lines:
            pred_heads, _, _ = self.parse(line)
            gold_heads = [token.head for token in line]
            acc += self.accuracy(pred_heads, gold_heads)
        return acc / len(lines)

    def evaluate_parallel(self, lines):
        acc = 0.0
        for line in lines:
            pred, _, _ = parse_parallel(
                line,
                self.arc_perceptron.weights,
                self.arc_perceptron.feature_dict,
                self.feature_opts,
                self.decoder)
            gold = [token.head for token in line]
            acc += self.accuracy(pred, gold)
        return acc / len(lines)

    def restore_from_parallel(self):
        self.arc_perceptron.restore_from_parallel()

    def average_weights(self):
        self.arc_perceptron.average_weights()

    def prune(self, eps):
        return self.arc_perceptron.prune(eps)

    def save(self, path, data_path=None, epochs_trained=None, accuracy=None):
        assert isinstance(data_path, str), data_path
        assert isinstance(epochs_trained, int), epochs
        assert isinstance(accuracy, dict), accuracy
        self.arc_accuracy = accuracy
        self.arc_perceptron.save(
            path, data=data_path, epochs=epochs_trained, accuracy=self.arc_accuracy)

    def load(self, path, training=False):
        accuracy, feature_opts = self.arc_perceptron.load(path, training)
        self.arc_accuracy, self.feature_opts = accuracy, feature_opts

    def top_features(self, n):
        return self.arc_perceptron.top_features(n)

    @property
    def weights(self):
        return self.arc_perceptron.weights
