import yaml

def load_config(config_file):
    """
    Load YAML configuration containing sample lists, marker flags, architecture pths, and folder structures. 
    
    Args:
        config_file (str): Filename containing paths to load

    Returns:
        config (dict): Parsed YAML config
        sample_dict (dict): Metadata per sample name
    """

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    vector_sample_name = [sample["name"] for sample in config["samples"]]   
    n_samples = len(vector_sample_name) # or len(sample_dict)
    sample_groups = [sample["group"] for sample in config["samples"]]
    n_groups = len(set(sample_groups))
    sample_ages = [sample["age"] for sample in config["samples"]]
    n_ages = len(set(sample_ages))

    print('n samples: ' + str(n_samples), flush=True)
    print('n groups: ' + str(n_groups), flush=True)
    print('n ages: ' + str(n_ages), flush=True)

    sample_dict = {s["name"]: s for s in config["samples"]}

    return config, sample_dict