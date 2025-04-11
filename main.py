import warnings
from src import get_config, train_model

def main():
    train_model()

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
