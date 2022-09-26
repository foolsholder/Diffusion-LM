from datasets import load_dataset


def download_glue(base_path):
    base_path = f"{base_path}/glue/"
    configs = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte']
    for config in configs:
        load_dataset("glue", config).save_to_disk(f"{base_path}/{config}")
    print(f"GLUE saved in {base_path}")


def download_super_glue(base_path):
    base_path = f"{base_path}/super_glue/"
    configs = ['copa', 'cb', 'boolq', 'wic', 'multirc', 'record']
    for config in configs:
        load_dataset("super_glue", config).save_to_disk(f"{base_path}/{config}")
    print(f"Super-GLUE saved in {base_path}")


def download_wiki_dpr(base_path):
    base_path = f"{base_path}/wiki_dpr/"
    load_dataset("wiki_dpr", "psgs_w100.multiset.no_index.no_embeddings").save_to_disk(base_path)


#base_path = "/home/vmeshchaninov/nlp_models/data"

#download_glue(base_path)
#download_super_glue(base_path)
#download_wiki_dpr(base_path)
