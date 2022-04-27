import json

import datasets
import csv
import torch
from datasets.tasks import QuestionAnsweringExtractive
from processdata.pre_process import get_label_dict
import random
logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

_URL = "../../oos/"
_URLS = {
    "train": _URL + "train.tsv",
    "dev": _URL + "dev.tsv",
    "test": _URL + "test.tsv",
    "neg": _URL + "squad.tsv",
}


class oodConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(oodConfig, self).__init__(**kwargs)


class Ood_data(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        oodConfig(
            name="ood data",
            version=datasets.Version("1.0.0", ""),
            description="ood data",
        ),
    ]

    def __init__(self, labels_dict,*args, **kwargs):

        super(Ood_data,self).__init__(*args, **kwargs)
        self.labels_dict=labels_dict
        self.labels_dict_keys=self.labels_dict.keys()

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.Value("int32"),
                "binary_label": datasets.Value("int32"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    # "filepath": downloaded_files["dev"],
                    "mode":"train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    # "filepath": downloaded_files["dev"],
                    "mode":"test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                    "mode":"val"
                },
            ),
        ]

    def _generate_examples(self, filepath ,mode):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        if mode=="train":
            with open(filepath, "r", encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                for idx, line in enumerate(reader):
                    if idx == 0:
                        continue
                    if line[1] not in self.labels_dict_keys:
                        continue
                    else:
                        label=self.labels_dict[line[1]]
                        binary_label=0
                    yield idx, {"text": line[0], "label": label,"binary_label":binary_label}
        elif mode=="val":
            with open(filepath, "r", encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                for idx, line in enumerate(reader):
                    if idx == 0:
                        continue
                    if line[1] not in self.labels_dict_keys:
                        continue
                    else:
                        label = self.labels_dict[line[1]]
                        binary_label = 0
                    yield idx, {"text": line[0], "label": label, "binary_label": binary_label}
        elif mode=="test" :
            with open(filepath, "r", encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                for idx, line in enumerate(reader):
                    if idx == 0:
                        continue
                    if line[1] not in self.labels_dict_keys:
                        label = len(self.labels_dict_keys)
                        binary_label = 1
                    else:
                        label = self.labels_dict[line[1]]
                        binary_label = 0

                    yield idx, {"text": line[0], "label": label, "binary_label": binary_label}
        else:
            raise ValueError("mode error")