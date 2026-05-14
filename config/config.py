import yaml

def load_config(config_path: str):
    """
    加载并解析 YAML 配置文件
    Args:
        config_path (str): 配置文件的路径
    Returns:
        dict: 解析后的配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
