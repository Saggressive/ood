import json

import datasets
import csv
import torch
from datasets.tasks import QuestionAnsweringExtractive

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
    "neg_train": _URL + "neg_train.tsv",
    "neg_val": _URL + "neg_val.tsv",
    "dev": _URL + "dev.tsv",
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
                    "filepath": downloaded_files["neg_train"],
                    # "filepath": downloaded_files["dev"],
                    "mode":"neg_train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["neg_val"],
                    # "filepath": downloaded_files["neg_train"],
                    # "filepath": downloaded_files["dev"],
                    "mode": "neg_val"
                },
            )
        ]

    def _generate_examples(self, filepath ,mode):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        if mode=="neg_train" or mode=="neg_val":
            with open(filepath, "r", encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                for idx, line in enumerate(reader):
                    if idx==0:
                        continue
                    if idx>=50000:
                        continue
                    label = len(self.labels_dict_keys)
                    binary_label = 1
                    yield idx, {"text": line[0], "label": label, "binary_label": binary_label}
        else:
            raise ValueError("mode error")