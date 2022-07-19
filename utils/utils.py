import os
import torch
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
import logging
import time
import datetime
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import random
import numpy as np
def get_logger(logs_dir):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s -  %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S')
    log_file_path = os.path.join(logs_dir, 'train.log')
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(log_format)
    handler2 = FileHandler(filename=log_file_path)
    handler2.setFormatter(log_format)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

def save_model(save_path,name,model,epoch,optimizer=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # assert os.path.exists(save_path)
    save_name = os.path.join(save_path,name)
    torch.save(model.state_dict(),save_name)
    torch.save({
        'epoch':epoch,
        'state_dict':model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimzer':optimizer.state_dict() if optimizer is not None else None
    },save_name)
    print(f"epoch{epoch+1}:the model is saved")
def load_model(path,model,optimizer=None):
    assert os.path.exists(path)
    save_dict = torch.load(path)
    epoch = save_dict['epoch']+1
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(save_dict['state_dict'])
    else:
        model.load_state_dict(save_dict['state_dict'])
    if save_dict['optimzier'] is not None and optimizer is not None:
        optimizer.load_state_dict(save_dict['optimzier'])
    return epoch

class Writer(object):

    def __init__(self,path,logname =None):
        if not os.path.exists(path):
            os.makedirs(path)
        if logname:
            save_name = os.path.join(path,logname)
        else:
            timestamp = datetime.datetime.now().strftime('%m%d-%H-%M')
            save_name = os.path.join(path,timestamp)
        self.writer = SummaryWriter(save_name)

    def add_scalars(self,flag,tag_values_pairs,step):

        for tag,value in tag_values_pairs.items():
            self.writer.add_scalar(flag+'/'+tag,value,step)

    def add_image(self,tag,tensor,step,format="nhw"):
        '''
        这个函数要求的是[0.1]的float32类型或者[0,255]的uint8类型
        :param tag:
        :param tensor:
        :param step:
        :param format: h * h* w    or 3 * h* w or nchw
        :return:
        '''
        ## the tensor is n* h* w
        # if len(tensor.size())==3：
        if format == 'nhw':
            tensor = tensor.unsqueeze(1).repeat(1,3,1,1)*255
        elif format =='3hw':
            tensor = tensor.unsqueeze(0)

            # ### the size is nchw
            # if tensor.size(0)>3:
        tensor = make_grid(tensor.to(torch.uint8),tensor.size(0))
        #  # to_num = tensor.astype(np.uint8)
        ### type一定要注意
        self.writer.add_image(tag,tensor,step,dataformats='CHW')
# zhe

def test_inference_time(model):
    device = torch.device("cuda")
    #model = Baseline(5, pretrained=True)
    #load_model("./save/lanenet_epoch_10.pth",model)
    model.to(device)

    imgs = torch.randn(5,200,3,360,640,dtype=torch.float).to(device)
    mask = torch.randn(1,1,360,640,dtype = torch.float).to(device)
    # label = torch.from_numpy(np.random.randint(2,size=(1,32,64)))
    starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    timing = np.zeros((196,1))
    model.eval()
    N, T, C, H, W = imgs.size()
    print(imgs.size())
    save_freq = 4
    for idx in range(N):
        frame = imgs[idx]  # the size is T * 3 * H * W
        features = []
        # pre-save
        token = None
        for t in range(save_freq):
            ## pure_data
            add = frame[t:t+1].repeat(save_freq-t,1,1,1)
            images = torch.cat([frame[:t+1],add],dim=0).unsqueeze(0)
            fea,token,output = model(images,mask)
            features.append(fea)
        # attention segment memory
        for t in range(save_freq, T):
            pre_features = features[-4:]
            starter.record()
            feature, token, output = model(frame[t:t + 1], masks=mask[0],pre_feature=pre_features, token=token)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timing[t-save_freq] += curr_time
            features.append(feature)
            features.pop(0)
    mean_syn = np.sum(timing) / ((T-4)*N)
    return mean_syn