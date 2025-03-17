import configparser


def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


if __name__ == "__main__":
    read_config("configs/config.yaml")
