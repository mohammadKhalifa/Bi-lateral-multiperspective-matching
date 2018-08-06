from data_loader import BLMPMDataLoader
from trainer import  Trainer
import torch
from model.model import *
from model.loss import *
from model.metric import *
from data_loader import MnistDataLoader, BLMPMDataLoader
from trainer import Trainer
import logging
import argparse
import json
import numpy as np
from base import BaseTrainer

CLASSIFICATION_THRESHOLD = 0.5

def test(config, model, test_data_loader):

    metrics = [eval(metric) for metric in config['metrics']]
    metrics_results= np.zeros([len(metrics)])

    for q, p, y in test_data_loader:
        
        output = model(q, p, test_data_loader.get_pretrained_embeddings())
        output = output.cpu().data.numpy()
        print (output)
        metrics_results+= np.array([metric(output, y) for metric in metrics])

        #print(metrics_results)



def main(config):
    
    # loading best model    
    model = eval(config['arch'])(config['model'])
    path = './saved/BLMPM/model_best.pth.tar'
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    data_loader = BLMPMDataLoader(config)

    test(config, model, data_loader)


    




if __name__ == '__main__':
    
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config',  type=str,
                        help='config file path (default: None)')

    args = parser.parse_args()
    config = json.load(open(args.config))
    
    config['test']= True # add test flag
    
    main(config)

    