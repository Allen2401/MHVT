import json
from torch.utils.data import DataLoader
from config import sys_config
from model.MHVT import MHVT
from utils.evaluator import Evaluator
from utils.utils import *
from config import Config

def train(model,train_loader,val_loader,val_dataset,start_epoch=0,end_epoch=100):
    setting = sys_config.get_train_setting()
    save_root = os.path.join(setting['result_dir'], setting['save_id'])
    writer = Writer(save_root)
    logger = get_logger(save_root)
    logger.info(json.dumps(setting,indent=4,ensure_ascii=False,sort_keys =False))

    for epoch in range(start_epoch,end_epoch):
        model.train()
        total_loss=0
        for index, batch_data in enumerate(train_loader):
            batch_data = [batch_data[i].cuda() for i in range(len(batch_data))]
            output,loss, indices = model(batch_data[0], batch_data[1].squeeze(-1), batch_data[2])
            optimizer.zero_grad()
            loss['total_loss'].backward()
            loss['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
            optimizer.step()
            total_loss +=loss['total_loss'].item()
            if index % setting['log_interval']==0:
                print(f"epoch {epoch}: {index}/{len(train_loader)-1} info: {loss}")
                writer.add_scalars("train",loss,epoch * len(train_loader)+index)
        logger.info(f"epoch {epoch}'s average loss is {total_loss/len(train_loader)}")

        if epoch % setting['save_interval']==0:
            save_model(save_root,f"epoch_{epoch}.pth", model,epoch,optimizer)
        if epoch !=0 and epoch % setting['val_interval'] == 0:
            model.eval()
            evaluator = Evaluator(val_dataset, exp_dir="")  # only metric
            val_total_loss = 0
            for i, data in enumerate(val_loader):
                idx = data[-1]
                data = [data[i].cuda() for i in range(len(data))]
                outputs, val_loss, val_indices = model(data[0], data[1].squeeze(-1), data[2])
                evaluator.add_prediction(idx,outputs)
                val_total_loss += val_loss['total_loss'].item()
            _, result = evaluator.eval()
            print(f"epoch {epoch} val result:{result}")
            result['loss'] = val_total_loss / len(val_loader)
            writer.add_scalars("val",result,epoch)
            logger.info(f"epoch {epoch}:val info is {result}")
        scheduler.step()

if __name__ == '__main__':
    seed_everything(42)
    sys_config = Config("./config/MHVT_Tusimple.yaml")
    setting = sys_config.get_train_setting()
    model = MHVT(**sys_config.get_model_parameters()).cuda()
    train_dataset = sys_config.get_dataset("train")
    val_dataset = sys_config.get_dataset('test')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)
    optimizer = sys_config.get_optimizer(model.parameters())
    scheduler = sys_config.get_lr_scheduler(optimizer)
    ##################################################################################################################################
    train(model,train_loader,val_loader,val_dataset=val_dataset,end_epoch=2501,start_epoch=0)



