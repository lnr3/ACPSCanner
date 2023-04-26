from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss,coverage_error

import torch
import numpy as np
import os

import gzip
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.PDB import PDBParser, MMCIFParser

def readname(file):
    i = 1
    name = []
    file = open(file, "r")
    for line in file:
        line = line.strip('\n')
        if i & 1:
            name.append(line)
        i += 1
    return name

def readfile(file1):
    i = 0
    number_of_label = 9

    file = open(file1, 'r')
    data = []
    tag = []
    for line in file:
        line = line.strip('\n')
        if i & 1:
            data.append(line)
        else:
            j = int(line[line.find('|') + 1 :])
            label = []
            label = [1 if (1 << 2) & j else 0] + label
            label = [1 if (1 << 7) & j else 0] + label
            label = [1 if (1 << 8) & j else 0] + label
            label = [1 if (1 << 10) & j else 0] + label
            label = [1 if (1 << 17) & j else 0] + label
            label = [1 if (1 << 18) & j else 0] + label
            label = [1 if (1 << 19) & j else 0] + label
            label = [1 if (1 << 28) & j else 0] + label
            label = [1 if (1 << 36) & j else 0] + label
            label.reverse()
            tag.append(label)
        i += 1
    return data, tag

def readstructure(data, file):
    strcuture = []
    for sequence in data:
        i = 0
        single = [0] * 250
        for ch in sequence:
            if ch == 'A':
                single[i] = 1
            elif ch == 'R':
                single[i] = 2
            elif ch == 'N':
                single[i] = 3
            elif ch == 'D':
                single[i] = 4
            elif ch == 'C':
                single[i] = 5
            elif ch == 'Q':
                single[i] = 6
            elif ch == 'E':
                single[i] = 7
            elif ch == 'G':
                single[i] = 8
            elif ch == 'H':
                single[i] = 9
            elif ch == 'I':
                single[i] = 10
            elif ch == 'L':
                single[i] = 11
            elif ch == 'K':
                single[i] = 12
            elif ch == 'M':
                single[i] = 13
            elif ch == 'F':
                single[i] = 14
            elif ch == 'P':
                single[i] = 15
            elif ch == 'S':
                single[i] = 16
            elif ch == 'T':
                single[i] = 17
            elif ch == 'W':
                single[i] = 18
            elif ch == 'Y':
                single[i] = 19
            elif ch == 'V':
                single[i] = 20
            i += 5
        strcuture.append(single)

    secondary = open(file, 'r')
    i = -1
    j = 0
    prev = ''
    for line in secondary:
        line = line.strip('\n')
        if len(line) < 1 or (line[0] != 'E' and line[0] != 'B'):
            continue
        vec = line.split('\t')
        if vec[2] != prev:
            i += 1
            j = 0
        prev = vec[2]
        strcuture[i][j + 1] = float(vec[-3])
        strcuture[i][j + 2] = float(vec[-2])
        strcuture[i][j + 3] = float(vec[-1])
        strcuture[i][j + 4] = float(vec[-6])
        j += 5
    return strcuture

def readesm(directory, file):
    files = readname(file)
    results = []
    for sequence in files:
        name = sequence[1:]
        cur = torch.load(directory + '/' + name + '.pt')['representations'][33]
        if cur.shape[0] < 50:
            cur = torch.cat((cur, torch.zeros(50 - cur.shape[0], 1280)), 0)
        results.append(cur)
    return results

def readpdb(directory, file):
    files = readname(file)
    results = []
    for pdb in files:
        name = pdb[1: ]
        map = StructureDataParser(directory + '/' + name + '.pdb', name)
        coords = map.generate_residue_coordinate()
        dist = euclidean_distances(coords, coords)
        # temp = coords.tolist()
        # while len(temp) < 50:
        #     temp.append([0, 0, 0])
        results.append(dist)
    return results

def euclidean_distances(x, y, squared=False):
    """
    Compute pairwise (squared) Euclidean distances.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    x_square = np.sum(x * x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y * y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances

class StructureDataParser:
    """
    PDB文件读取
    """

    def __init__(self, path, protein_id, file_type='pdb'):
        """
        init function
        :param path: file path
        :param protein_id: protein id
        :param file_type: Type of the file to be read，the file type can be pdb or mmcif
        """
        if path.endswith('gz'):
            self.path = gzip.open(path, "rt", encoding='UTF-8')
        else:
            self.path = open(path, "rt", encoding='UTF-8')
        self.protein_id = protein_id
        self.file_type = file_type
        if self.file_type == 'pdb':
            self.parser = PDBParser()
        elif self.file_type == 'mmcif':
            self.parser = MMCIFParser()
        else:
            raise ValueError(f"{file_type} is not pdb or mmcif")
        self.structure = self.parser.get_structure(self.protein_id, self.path)
        self.sequence = self.get_sequence()
        self.sequence_len = len(self.sequence)

    def generate_atom_distance_map(self, atom='CA'):
        coords_ = self.get_residue_coordinates(atom)
        return euclidean_distances(coords_, coords_)

    def get_residues(self):
        return [res for res in self.structure.get_residues()]

    def get_sequence(self):
        return [protein_letters_3to1[res.resname] for res in self.structure.get_residues()]

    def get_residue_coordinates(self, atom='CA'):
        return self.generate_residue_coordinate(atom)

    def generate_residue_coordinate(self, atom='CA'):
        coord_list = [res[atom].coord for res in self.structure.get_residues()]
        coords_ = np.stack(coord_list)
        return coords_

    def get_residue_atoms_coords(self, atoms=None):
        if atoms is None:
            atoms = ['CA', 'C', 'N']
        coords = {atom: self.get_residue_coordinates('CA').tolist() for atom in atoms}
        return coords


def accuracy(y_true, y_pred):
    count = 0
    for i in range(len(y_true)):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += 0 if q == 0 else p / q
    return count / len(y_true)

def one_error(y_true, y_pred):
    N = len(y_true)
    label_index = []
    for i in range(N):
        index = np.where(y_true[i] == 1)[0]
        label_index.append(index)
    OneError = 0
    for i in range(N):
        if np.argmax(y_pred[i]) not in label_index[i]:
            OneError += 1
    return OneError / N

def metrics(y_true, y_pred, y_pred_prob):
    emr = accuracy_score(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='samples')
    recall = recall_score(y_true, y_pred, average='samples')
    f1 = f1_score(y_true, y_pred, average='samples')
    hloss = hamming_loss(y_true, y_pred)
    coverage = coverage_error(y_true, y_pred_prob) - 1
    return emr, acc, precision, recall, f1, hloss, coverage
