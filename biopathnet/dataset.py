import os
import csv
import glob
from tqdm import tqdm
from ogb import linkproppred

import torch
import torch.nn as tnn
from torch.utils import data as torch_data

from torchdrug import data, datasets, utils
from torchdrug.core import Registry as R


class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

    def load_inductive_tsvs(self, train_files, test_files, verbose=0):
        assert len(train_files) == len(test_files)
        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        train_entity_vocab, inv_train_entity_vocab = self._standarize_vocab(None, inv_train_entity_vocab)
        test_entity_vocab, inv_test_entity_vocab = self._standarize_vocab(None, inv_test_entity_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        self.train_graph = data.Graph(triplets[:num_samples[0]],
                                      num_node=len(train_entity_vocab), num_relation=len(relation_vocab))
        self.valid_graph = self.train_graph
        self.test_graph = data.Graph(triplets[sum(num_samples[:2]): sum(num_samples[:3])],
                                     num_node=len(test_entity_vocab), num_relation=len(relation_vocab))
        self.graph = self.train_graph
        self.triplets = torch.tensor(triplets[:sum(num_samples[:2])] + triplets[sum(num_samples[:3]):])
        self.num_samples = num_samples[:2] + num_samples[3:]
        self.train_entity_vocab = train_entity_vocab
        self.test_entity_vocab = test_entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_train_entity_vocab = inv_train_entity_vocab
        self.inv_test_entity_vocab = inv_test_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def __getitem__(self, index):
        return self.triplets[index]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.CoraLinkPrediction")
class CoraLinkPrediction(datasets.Cora):

    def __init__(self, **kwargs):
        super(CoraLinkPrediction, self).__init__(**kwargs)
        self.transform = None

    def __getitem__(self, index):
        return self.graph.edge_list[index]

    def __len__(self):
        return self.graph.num_edge

    def split(self, ratios=(85, 5, 10)):
        length = self.graph.num_edge
        norm = sum(ratios)
        lengths = [int(r / norm * length) for r in ratios]
        lengths[-1] = length - sum(lengths[:-1])

        g = torch.Generator()
        g.manual_seed(0)
        return torch_data.random_split(self, lengths, generator=g)


@R.register("datasets.CiteSeerLinkPrediction")
class CiteSeerLinkPrediction(datasets.CiteSeer):

    def __init__(self, **kwargs):
        super(CiteSeerLinkPrediction, self).__init__(**kwargs)
        self.transform = None

    def __getitem__(self, index):
        return self.graph.edge_list[index]

    def __len__(self):
        return self.graph.num_edge

    def split(self, ratios=(85, 5, 10)):
        length = self.graph.num_edge
        norm = sum(ratios)
        lengths = [int(r / norm * length) for r in ratios]
        lengths[-1] = length - sum(lengths[:-1])

        g = torch.Generator()
        g.manual_seed(0)
        return torch_data.random_split(self, lengths, generator=g)


@R.register("datasets.PubMedLinkPrediction")
class PubMedLinkPrediction(datasets.PubMed):

    def __init__(self, **kwargs):
        super(PubMedLinkPrediction, self).__init__(**kwargs)
        self.transform = None

    def __getitem__(self, index):
        return self.graph.edge_list[index]

    def __len__(self):
        return self.graph.num_edge

    def split(self, ratios=(85, 5, 10)):
        length = self.graph.num_edge
        norm = sum(ratios)
        lengths = [int(r / norm * length) for r in ratios]
        lengths[-1] = length - sum(lengths[:-1])

        g = torch.Generator()
        g.manual_seed(0)
        return torch_data.random_split(self, lengths, generator=g)


@R.register("datasets.FB15k237Inductive")
class FB15k237Inductive(InductiveKnowledgeGraphDataset):

    train_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
    ]

    test_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        train_files = []
        for url in self.train_urls:
            url = url % version
            save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            train_files.append(txt_file)
        test_files = []
        for url in self.test_urls:
            url = url % version
            save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            test_files.append(txt_file)

        self.load_inductive_tsvs(train_files, test_files, verbose=verbose)


@R.register("datasets.WN18RRInductive")
class WN18RRInductive(InductiveKnowledgeGraphDataset):

    train_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
    ]

    test_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        train_files = []
        for url in self.train_urls:
            url = url % version
            save_file = "wn18rr_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            train_files.append(txt_file)
        test_files = []
        for url in self.test_urls:
            url = url % version
            save_file = "wn18rr_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            test_files.append(txt_file)

        self.load_inductive_tsvs(train_files, test_files, verbose=verbose)


@R.register("datasets.OGBLBioKG")
class OGBLBioKG(data.KnowledgeGraphDataset):

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        self.path = path

        dataset = linkproppred.LinkPropPredDataset("ogbl-biokg", path)
        self.load_ogb(dataset, verbose=verbose)

    def load_ogb(self, dataset, verbose=1):
        entity_vocab = []
        relation_vocab = []
        entity_type_vocab = []
        inv_entity_type_offset = {}
        entity_type2num = []

        zip_files = glob.glob(os.path.join(dataset.root, "mapping/*.gz"))
        for zip_file in zip_files:
            csv_file = utils.extract(zip_file)
            type = os.path.basename(csv_file).split("_")[0]
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin)
                if verbose:
                    reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
                fields = next(reader)
                if "relidx" in csv_file:
                    for index, token in reader:
                        relation_vocab.append(token)
                else:
                    entity_type_vocab.append(type)
                    inv_entity_type_offset[type] = len(entity_vocab)
                    num_entity = 0
                    for index, token in reader:
                        entity_vocab.append("%s (%s)" % (type, token))
                        num_entity += 1
                    entity_type2num.append(num_entity)

        edge_split = dataset.get_edge_split()
        triplets = []
        num_samples = []
        num_samples_with_neg = []
        negative_heads = []
        negative_tails = []
        for key in ["train", "valid", "test"]:
            split_dict = edge_split[key]
            h = torch.as_tensor(split_dict["head"])
            t = torch.as_tensor(split_dict["tail"])
            r = torch.as_tensor(split_dict["relation"])
            h_type = torch.tensor([inv_entity_type_offset[h] for h in split_dict["head_type"]])
            t_type = torch.tensor([inv_entity_type_offset[t] for t in split_dict["tail_type"]])

            h = h + h_type
            t = t + t_type
            triplet = torch.stack([h, t, r], dim=-1)
            triplets.append(triplet)
            num_samples.append(len(h))
            if "head_neg" in split_dict:
                neg_h = torch.as_tensor(split_dict["head_neg"])
                neg_t = torch.as_tensor(split_dict["tail_neg"])
                neg_h = neg_h + h_type.unsqueeze(-1)
                neg_t = neg_t + t_type.unsqueeze(-1)
                negative_heads.append(neg_h)
                negative_tails.append(neg_t)
                num_samples_with_neg.append(len(h))
            else:
                num_samples_with_neg.append(0)
        triplets = torch.cat(triplets)

        self.load_triplet(triplets, entity_vocab=entity_vocab, relation_vocab=relation_vocab)
        entity_type_vocab, inv_entity_type_vocab = self._standarize_vocab(entity_type_vocab, None)
        self.entity_type_vocab = entity_type_vocab
        self.inv_entity_type_vocab = inv_entity_type_vocab
        self.num_samples = num_samples
        self.num_samples_with_neg = num_samples_with_neg
        self.negative_heads = torch.cat(negative_heads)
        self.negative_tails = torch.cat(negative_tails)

        node_type = []
        for i, num_entity in enumerate(entity_type2num):
            node_type += [i] * num_entity
        with self.graph.node():
            self.graph.node_type = torch.tensor(node_type)

    def split(self, test_negative=True):
        offset = 0
        neg_offset = 0
        splits = []
        for num_sample, num_sample_with_neg in zip(self.num_samples, self.num_samples_with_neg):
            if test_negative and num_sample_with_neg:
                pos_h, pos_t, pos_r = self[offset: offset + num_sample].t()
                neg_h = self.negative_heads[neg_offset: neg_offset + num_sample_with_neg]
                neg_t = self.negative_tails[neg_offset: neg_offset + num_sample_with_neg]
                num_negative = neg_h.shape[-1]
                h = pos_h.unsqueeze(-1).repeat(2, num_negative + 1)
                t = pos_t.unsqueeze(-1).repeat(2, num_negative + 1)
                r = pos_r.unsqueeze(-1).repeat(2, num_negative + 1)
                t[:num_sample_with_neg, 1:] = neg_t
                h[num_sample_with_neg:, 1:] = neg_h
                split = torch.stack([h, t, r], dim=-1)
            else:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
            neg_offset += num_sample_with_neg
        return splits

@R.register("datasets.biomedical")
class biomedical(data.KnowledgeGraphDataset):
    """
    load training, validation and testing triplets from multiple files
    """

    files = [
        "train1.txt", # BRG
        "train2.txt",  # training triplets
        "valid.txt", # validation triplets
        "test.txt",] # test triplets

    entity_files = ['entity_types.txt', 'entity_names.txt']

    def __init__(self, path, include_factgraph=True, fact_as_train=False, verbose=1, files=None, entity_files = None):
        if files:
            self.files = files
            
        if entity_files:
            self.entity_files = entity_files
            
        path = os.path.expanduser(path)
        self.path = path
        self.include_factgraph = include_factgraph
        self.fact_as_train = fact_as_train
        
        chosen_files = self.files if self.include_factgraph else self.files[1:]

        txt_files=[]
        for x in chosen_files:
            txt_files.append(os.path.join(self.path, x))

        self.load_tsvs(txt_files, verbose=verbose)
        self.load_entity_types(path)

    def load_tsvs(self, tsv_files, verbose=0):
        """
        Load the dataset from multiple tsv files.

        Parameters:
            tsv_files (list of str): list of file names
            verbose (int, optional): output verbose level
        """
        inv_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for tsv_file in tsv_files:
            with open(tsv_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % tsv_file, utils.get_line_count(tsv_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_entity_vocab:
                        inv_entity_vocab[h_token] = len(inv_entity_vocab)
                    h = inv_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_entity_vocab:
                        inv_entity_vocab[t_token] = len(inv_entity_vocab)
                    t = inv_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)
        self.num_samples = num_samples
        
         
    def load_triplet(self, triplets, entity_vocab=None, relation_vocab=None, inv_entity_vocab=None,
                     inv_relation_vocab=None):
        """
        Load the dataset from triplets.
        The mapping between indexes and tokens is specified through either vocabularies or inverse vocabularies.

        Parameters:
            triplets (array_like): triplets of shape :math:`(n, 3)`
            entity_vocab (dict of str, optional): maps entity indexes to tokens
            relation_vocab (dict of str, optional): maps relation indexes to tokens
            inv_entity_vocab (dict of str, optional): maps tokens to entity indexes
            inv_relation_vocab (dict of str, optional): maps tokens to relation indexes
        """
        entity_vocab, inv_entity_vocab = self._standarize_vocab(entity_vocab, inv_entity_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(relation_vocab, inv_relation_vocab)

        num_node = len(entity_vocab) if entity_vocab else None
        num_relation = len(relation_vocab) if relation_vocab else None
        self.graph = data.Graph(triplets, num_node=num_node, num_relation=num_relation)
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_entity_vocab = inv_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def load_entity_types(self, path) -> None:
        inv_type_vocab = {}
        node_type = {}
        # read in node types
        with open(os.path.join(path, self.entity_files[0]), "r") as f:
            lines = f.readlines()
            for line in lines:
                entity_token, type_token = line.strip().split()
                if type_token not in inv_type_vocab:
                    inv_type_vocab[type_token] = len(inv_type_vocab)
                if entity_token in self.inv_entity_vocab:
                    node_type[self.inv_entity_vocab[entity_token]] = inv_type_vocab[type_token]
        assert len(node_type) == self.num_entity
        _, node_type = zip(*sorted(node_type.items()))
        with self.graph.node():
            self.graph.node_type = torch.tensor(node_type)

    def split(self):
        offset = 0
        splits = []
        num_samples = self.num_samples
        if self.include_factgraph and self.fact_as_train:
            num_samples = [num_samples[0] + num_samples[1]] + num_samples[2:]
        for num_sample in num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
            
        if self.include_factgraph and not self.fact_as_train:
            return splits[1:]
        else:
            return splits
        
    def get_fact1(self):
        offset = 0
        splits = []
        num_samples = self.num_samples
        if self.include_factgraph and self.fact_as_train:
            num_samples = [num_samples[0] + num_samples[1]] + num_samples[2:]
        for num_sample in num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        if self.include_factgraph:
            return splits[0]
        else:
            return None

@R.register("datasets.BiomedicalInductive")
class BiomedicalInductive(data.KnowledgeGraphDataset):
    """
    Data class for inductive reasoning in biomedical knowledge graphs.

    This class is designed to support inductive link prediction, where the 
    training and inference graphs are disjoint. It is inherited from the 
    `KnowledgeGraphDataset` base class.

    Attributes:
        train_graph (Graph): The graph used during training, built using train1.txt (BRG - if available) and train2.txt.
        valid_graph (Graph): The same as train_graph.
        test_graph (Graph): The graph used during testing time, build using test_graph.txt.
        train_entity_vocab (Dict[str, int]): Vocabulary mapping of training graph mapping for biomedical entities.
        test_entity_vocab (Dict[str, int]): Vocabulary mapping of training graph mapping for biomedical entities.
        relation_vocab (Dict[str, int]): Shared vocabulary mapping for biomedical relations.
        inv_train_entity_vocab (Dict[int, str]): Inverse vocabulary mapping.


    Methods:
        load_inductive_tsvs(): Loads the biomedical files and creates the different graphs and vocabularies.
        load_entity_types(): Loads the entity types of nodes from all files.
    """
    
    files = [
        "train1.txt", # BRG
        "train2.txt",  # training triplets
        "valid.txt", # validation triplets
        "test_graph.txt", # inference graph
        "test.txt",  # test triplets
    ]

    entity_files = ['entity_types.txt', 'entity_names.txt'] # entity types and names from training graph and test graph
    
    def __init__(self, path, verbose=1, files=None, entity_files=None):
        if files:
            self.files = files
            
        if entity_files:
            self.entity_files = entity_files
            
        path = os.path.expanduser(path)
        self.path = path
        
        txt_files=[]
        for x in self.files:
            txt_files.append(os.path.join(self.path, x))
        
        self.load_inductive_tsvs(txt_files[:-2], txt_files[-2:], verbose=verbose)
        self.load_entity_types(path)

    def load_inductive_tsvs(self, train_files, test_files, verbose=0) -> None:
        """
        Load training and inference data from TSV files for inductive reasoning.

        This function reads TSV files from two lists: one denoting the files for
        training and another for the testing. Files for training include train1.txt, 
        train2.txt, with the validation triplets on the training graph in valid.txt. 
        Files for testing are test_graph.txt and test.txt.  Further, the training and 
        testing entity vocabularies are built, as well as the shared relation vocabulary.
        """
    
        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        train_entity_vocab, inv_train_entity_vocab = self._standarize_vocab(None, inv_train_entity_vocab)
        test_entity_vocab, inv_test_entity_vocab = self._standarize_vocab(None, inv_test_entity_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        self.train_graph = data.Graph(triplets[:sum(num_samples[:2])], # train1 + train2 
                                      num_node=len(train_entity_vocab), num_relation=len(relation_vocab))
        self.valid_graph = self.train_graph
        self.test_graph = data.Graph(triplets[sum(num_samples[:3]): sum(num_samples[:4])], # 4th file - test_graph
                                     num_node=len(test_entity_vocab), num_relation=len(relation_vocab))
        self.graph = self.train_graph
     
        self.triplets = torch.tensor(triplets[:sum(num_samples[0:3])] # train2 + valid
                                         + triplets[sum(num_samples[:-1]):]) # test
        self.num_samples = num_samples[:3] + num_samples[4:]
        # self.triplets = torch.tensor(triplets[num_samples[0]:sum(num_samples[:3])] # train2 + valid
        #                                  + triplets[sum(num_samples[:-1]):]) # test
        # self.num_samples = num_samples[1:3] + num_samples[4:] 
        
        self.train_entity_vocab = train_entity_vocab
        self.test_entity_vocab = test_entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_train_entity_vocab = inv_train_entity_vocab
        self.inv_test_entity_vocab = inv_test_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def load_entity_types(self, path) -> None:
        inv_train_type_vocab = {}
        inv_test_type_vocab = {}
        node_type_train = {}
        node_type_test = {}
        # read in node types
        with open(os.path.join(path, self.entity_files[0]), "r") as f:
            lines = f.readlines()
            for line in lines:
                entity_token, type_token = line.strip().split()
                if type_token not in inv_train_type_vocab:
                    inv_train_type_vocab[type_token] = len(inv_train_type_vocab)
                if type_token not in inv_test_type_vocab:
                    inv_test_type_vocab[type_token] = len(inv_test_type_vocab)
                if entity_token in self.inv_train_entity_vocab:
                    node_type_train[self.inv_train_entity_vocab[entity_token]] = inv_train_type_vocab[type_token]
                if entity_token in self.inv_test_entity_vocab:
                    node_type_test[self.inv_test_entity_vocab[entity_token]] = inv_test_type_vocab[type_token]

        assert self.test_graph.num_node == len(node_type_test)
        assert self.train_graph.num_node == len(node_type_train)
        _, node_type_train = zip(*sorted(node_type_train.items()))
        _, node_type_test = zip(*sorted(node_type_test.items()))

        # train_graph, valid_graph and graph are all the same
        for g in [self.train_graph, self.graph, self.valid_graph]:
            with g.node():
                g.node_type = torch.tensor(node_type_train)
        # test_graph
        with self.test_graph.node():
            self.test_graph.node_type = torch.tensor(node_type_test)

    def __getitem__(self, index: int):
        return self.triplets[index]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits[1:]