import os
import json
import functools
from datasets import load_dataset, load_from_disk, IterableDataset, disable_progress_bar
from datasets.utils.logging import set_verbosity_error

from . import load
from .preprocessing import text_preprocessor, unsupervised_preprocessor
from .dataset import IterableDataset

disable_progress_bar()
set_verbosity_error()


def create_c4_dataset(split):
    import os
    base_path = os.environ['C4_PATH']
    #base_path = "/home/gbartosh/nlp_models/data/c4/en"
    if split == 'valid':
        split = 'validation'
    data_files = {
        # "train": [f"{base_path}/c4-train.{idx:05d}-of-01024.json.gz" for idx in range(1024)],
        # "validation": [f"{base_path}/c4-validation.{idx:05d}-of-00008.json.gz" for idx in range(8)],
        "validation": f"{base_path}/c4-validation.*.json.gz",
        "train": f"{base_path}/c4-train.0000*.json.gz",
    }
    return {"c4": load_dataset(path=base_path, data_files=data_files, split=split)}


def create_wiki_dpr_dataset(split):
    base_path = "/home/vmeshchaninov/nlp_models/data"
    if not os.path.isdir(f"{base_path}/wiki_dpr"):
        load.download_wiki_dpr(base_path)
    return {"wiki_dpr": load_from_disk(f"{base_path}/wiki_dpr").get(split)}


def create_glue_dataset(split):
    base_path = "/home/vmeshchaninov/nlp_models/data/glue/"
    configs = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte']
    return {config: load_from_disk(f"{base_path}/{config}").get(split) for config in configs}


def create_super_glue_dataset(split):
    base_path = "/home/vmeshchaninov/nlp_models/data/super_glue/"
    configs = ['copa', 'cb', 'boolq', 'wic', 'multirc', 'record']
    return {config: load_from_disk(f"{base_path}/{config}").get(split) for config in configs}


def create_unsupervised_dataset(split, weights_sampling_mode="size", config_path: str = "config.json", tokenizer=None,
                                max_sequence_len=64):
    with open(config_path, "rb") as file:
        config = json.load(file)
    #
    # datasets = dict()
    # datasets.update(create_c4_dataset(split))
    # # datasets.update(create_wiki_dpr_dataset(split))
    # # datasets.update(create_glue_dataset(split))
    # # datasets.update(create_super_glue_dataset(split))
    #
    # for name in list(datasets.keys()):
    #     dt = datasets[name]
    #     if dt is None:
    #         datasets.pop(name)
    #         continue
    #     datasets[name] = text_preprocessor(dt=dt, config=config, benchmark_name=name)
    #     if name not in ["c4"]:
    #         datasets[name] = unsupervised_preprocessor(datasets[name], benchmark_name=name)
    #     print(f"{name} is loaded")
    #
    # max_length = config["model"]["max_length"]
    # return IterableDataset(datasets, max_length=max_length, config=config, weights_sampling_mode=weights_sampling_mode)
    dt = create_c4_dataset(split)["c4"]
    dt = text_preprocessor(
        dt=dt,
        config=config,
        benchmark_name="c4"
    )
    dt.set_transform(lambda x: tokenizer(x["inputs"],
                                         max_length=max_sequence_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt"))
    return dt
