import torch
from utils.evaluator import Evaluator
from model.MHVT import MHVT
from torch.utils.data import DataLoader
from dataset.LaneDataset import LaneDataset
from config import Config
if __name__ == '__main__':
    sys_config = Config("config/MHVT_Tusimple.yaml")
    setting = sys_config.get_train_setting()
    model = MHVT(**sys_config.get_model_parameters()).cuda()
    model.load_state_dict(torch.load("./save/Ours_best.pth")['state_dict'])
    dataset = LaneDataset(split="test")
    loader = DataLoader(dataset,batch_size = 1,shuffle=False,num_workers=2)
    evaluator = Evaluator(dataset, exp_dir="")
    model.eval()
    for index,data in enumerate(loader):
        idx = data.pop()
        data = [part.cuda() for part in data]
        outputs, val_loss, val_indices = model(data[0], data[1].squeeze(-1), data[2])
        evaluator.add_prediction(idx, outputs)
        if index % 10==0:
            print(f"has handled {index} pics")
    print(evaluator.eval())
