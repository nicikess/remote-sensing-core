from typing import Any, Callable, Dict

from lit_diffusion.util import instantiate_python_class_from_string_config


def gather_stats(dataset_config: Dict, metrics: Dict[str, Callable]) -> Dict[str, Any]:
    dataset = instantiate_python_class_from_string_config(class_config=dataset_config)
    sample_sample = dataset[0]
    result_dict = {
        k: {
            metrics_name: metric(sample_sample)
            for metrics_name, metric in metrics.items()
        }
        for k in sample_sample.keys()
    }
    for sample in dataset:
        for k, v in sample:
            for metrics_name, metric in metrics.items():
                global_value = result_dict[k][metrics_name]
                result_dict[k][metrics_name] = metric([metric(v), global_value])
    return result_dict


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    import numpy as np

    config_path = Path("../configs/stats/ben-ge-100.yaml")
    with config_path.open("r") as config_file:
        dataset_config = yaml.safe_load(config_file)["dataset_config"]

    metrics = {
        "maximum": np.min,
        "minimum": np.maximum,
    }
    result = gather_stats(dataset_config=dataset_config, metrics=metrics)
    print(result)
    with Path("./output.yaml", "w") as output_file:
        yaml.safe_dump(data=result, stream=output_file)
