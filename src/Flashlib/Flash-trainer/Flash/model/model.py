import torch
from tqdm import tqdm 
import numpy as np 
import neptune

from torch.optim.swa_utils import AveragedModel
from Flash.utils import update_bn

class Model(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self.optimizer = None
        self.scheduler = None
        self.trainloader = None
        self.validloader = None
        self.current_epoch = None
        self.SWA = None
        self.swa_model = None
        self.swa_start = None
        self.device = None
        self.trainloader = None 
        self.validloader = None
        self.epochs = None
        self.logger = None

    def forward(self, *args, **kwargs):
        super.forward(*args, **kwargs)

    def configure_optimizer(self, *args, **kwargs):
        return 
    
    def configure_scheduler(self, *args, **kwargs):
        return 
    
    def compute_on_end_metrics(self, outputs, targets):
        return

    def train_one_step(self, data, device, fp16 = False, scaler = None):
        
        for k,v in data.items():
            data[k] = v.to(device)

        if fp16:
            with torch.cuda.amp.autocast():
                self.optimizer.zero_grad()

                _, loss, metric = self(**data)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                return _,loss, metric
        else:
            self.optimizer.zero_grad()
            _,loss,metric = self(**data)
            loss.backward()
            self.optimizer.step()
            return _,loss, metric

    def train_one_epoch(self, data_loader, device, fp16 = None, on_end_metrics = None, logger = None):
        self.train()
        fin_loss = 0.0
        monitor = {}
        temp = {}
        if on_end_metrics == True:
            outputs = []
            targets = []
        tk = tqdm(data_loader, desc = "Epoch" + " [TRAIN] " + str(self.current_epoch+1))
        if fp16 == True:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        for t, data in enumerate(tk):
            logits,loss,metric = self.train_one_step(data, device, fp16, scaler)
            fin_loss += loss.item()
            if metric != None:
                if t == 0 :
                    for k, v in metric.items():
                        temp[k] = v 
                else:
                    for k,v in metric.items():
                        temp[k] += v 
                    for k,v in temp.items():
                        monitor[str('trn_live_')+k] = v / (t + 1)
    
                monitor['trn_live_loss'] = (fin_loss) / (t+1)
                monitor['LR'] = self.optimizer.param_groups[0]['lr']
                tk.set_postfix(monitor)

                if logger == True:
                    for k,v in monitor.items():
                        neptune.log_metric(k, v)
            else:
                tk.set_postfix({'trn_live_loss' : '%.6f' %float(fin_loss / (t+1)), 'LR' : self.optimizer.param_groups[0]['lr']})
                
                if logger == True:
                    neptune.log_metric('trn_live_loss', (fin_loss / (t+1)))
                    neptune.log_metric('LR', self.optimizer.param_groups[0]['lr'])

            if on_end_metrics == True:
                outputs += logits
                targets += data[self.target_name]
    
        if self.scheduler != None:
            self.scheduler.step()

        if self.swa_model:
            if self.current_epoch >= self.swa_start:
                self.swa_model.update_parameters(self)
                print("SWA is being used")
                if self.scheduler != None:
                    self.scheduler.step()

        if on_end_metrics == True:
            return fin_loss / len(data_loader), outputs, targets 

        return fin_loss / len(data_loader), None, None 

    def valid_one_step(self, data, device):
        for k,v in data.items():
            data[k] = v.to(device)
        _, loss, metric = self(**data)
        return _, loss, metric

    def valid_one_epoch(self, data_loader, device, on_end_metrics = None, logger = None):
        self.eval()
        fin_loss = 0.0
        monitor = {}
        temp = {}
        if on_end_metrics == True:
            outputs = []
            targets = []
        tk = tqdm(data_loader, desc = "Epoch" + " [VALID] " + str(self.current_epoch+1))
    
        for t, data in enumerate(tk):

            with torch.no_grad():
                logits,loss,metric = self.valid_one_step(data, device)
            fin_loss += loss.item()
            if metric != None:
                if t == 0 :
                    for k, v in metric.items():
                        temp[k] = v 
                else:
                    for k,v in metric.items():
                        temp[k] += v 
                    for k,v in temp.items():
                        monitor[str('val_live_')+k] = v / (t + 1)
                    
                monitor['val_live_loss'] = (fin_loss) / (t+1)
                tk.set_postfix(monitor)

                if logger == True:
                    for k,v in monitor.items():
                        neptune.log_metric(k, v)
            else:
                tk.set_postfix({'val_live_loss' : '%.6f' %float(fin_loss / (t+1))})

                if logger == True:
                    neptune.log_metric('val_live_loss', (fin_loss / (t+1)))
                
            if on_end_metrics == True:
                outputs += logits
                targets += data[self.target_name]
            
        if on_end_metrics == True:
            return fin_loss / len(data_loader), outputs, targets 

        return fin_loss / len(data_loader), None, None
    
    def get_swa_model(self):
        print("Updating batch norm parameters for SWA return....")
        update_bn(self.trainloader, self.swa_model, device = self.device)
        return self.swa_model

    def fit(
        self,
        trainset,
        validset,
        device = 'cuda',
        epochs = 1,
        batch_size = 32,
        fp16 = True,
        train_shuffle = True,
        valid_shuffle = False,
        drop_last_batch_train = True,
        pin_memory = False,
        num_workers = 0,
        on_end_metrics = False,
        save_model_based_on_loss = True,
        save_model_based_on_metric = False,
        based_on_metric = None,
        target_name = None,
        metric_order = 'greater_than',
        model_path = None,
        logger = True,
        neptune_api_token = None,
        neptune_init = None,
        neptune_experiment_name = None,
        neptune_params = None, 
        SWA = False,
        swa_start = 0
    ):

        '''
            trainset : training set 
            validset : validation set 
            device : specidy device
            epochs : no. of epochs
            batch_size : no. of batch size
            fp16 : transfer to float16
            train_shuffle : shuffle for trainloader
            valid_shuffle : shuffle for validloader
            drop_last_batch_train : drop the last batch 
            pin_memory = specify pin_memory (True, False)
            num_workers : specify num 
            on_end_metrics : if compute_on_end_metrics()
            save_model_based_on_loss : save based on loss
            save_model_based_on_metric : save based on metric 
            model_path : name with path 
            logger : default neptune (True, False)
            neptune_logger info
        '''
        
        self.device = device
        self.epochs = epochs
        self.logger = logger 
        self.SWA = SWA
        self.swa_start = swa_start
        self.target_name = target_name

        if self.logger == True:
            neptune.init(neptune_init, neptune_api_token)
            neptune.create_experiment(neptune_experiment_name, params = neptune_params)


        if self.trainloader is None:
            self.trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size = batch_size,
                shuffle = train_shuffle,
                drop_last = drop_last_batch_train,
                pin_memory = pin_memory,
                num_workers = num_workers
            )
        
        if validset:
            if self.validloader is None:
                self.validloader = torch.utils.data.DataLoader(
                    validset,
                    batch_size = batch_size,
                    shuffle = valid_shuffle,
                    pin_memory = pin_memory,
                    num_workers = num_workers
                )

        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

        if self.SWA:
            self.swa_model = AveragedModel(self)
            print("SWA Initialized")

        loss_best_score = np.Inf
        if metric_order == 'less_than':
            metric_best_score = np.Inf
        elif metric_order == 'greater_than':
            metric_best_score = 0.0
        temp_tr_metrics = {}
        temp_vl_metrics = {}

        for i in range(epochs):
            self.current_epoch = i
            train_loss, tr_outputs, tr_targets = self.train_one_epoch(self.trainloader, device, fp16, on_end_metrics = on_end_metrics, logger = logger)
            if validset:
                valid_loss, vl_outputs, vl_targets = self.valid_one_epoch(self.validloader, device, on_end_metrics = on_end_metrics, logger = logger)

            if on_end_metrics:
                tr_end_metrics = self.compute_on_end_metrics(tr_outputs, tr_targets)
                if validset:
                    vl_end_metrics = self.compute_on_end_metrics(vl_outputs, vl_targets)

                for k,v in tr_end_metrics.items():
                    temp_tr_metrics[str('train_')+k] = v
                
                if validset:
                    for k,v in vl_end_metrics.items():
                        temp_vl_metrics[str('valid_')+k] = v

            if logger == True:
                neptune.log_metric('per_epoch_train_loss', train_loss)
                if validset:
                   neptune.log_metric('per_epoch_valid_loss', valid_loss)
                if on_end_metrics:
                    for k,v in temp_tr_metrics.items():
                        neptune.log_metric(k,v)
                    if validset:
                        for k, v in temp_vl_metrics.items():
                            neptune.log_metric(k,v)
                


            if validset:
                if save_model_based_on_loss == True:
                    if valid_loss < loss_best_score:
                        save_utils = {'model' : self.state_dict(), 'best_score' : loss_best_score, 'epochs' : i+1}
                        torch.save(save_utils, 'Based-on-loss-' + model_path)
                        print('MODEL_SAVED')
                        loss_best_score = valid_loss

                if save_model_based_on_metric == True:
                    if metric_order == 'less_than':
                        if vl_end_metrics[based_on_metric] < metric_best_score:
                            save_utils = {'model' : self.state_dict(), 'best_score' : loss_best_score, 'epochs' : i+1}
                            torch.save(save_utils, model_path)
                            print('MODEL_SAVED')
                            metric_best_score = vl_end_metrics[based_on_metric]

                    elif metric_order == 'greater_than':
                        if vl_end_metrics[based_on_metric] > metric_best_score:
                            save_utils = {'model' : self.state_dict(), 'best_score' : loss_best_score, 'epochs' : i+1}
                            torch.save(save_utils, model_path)
                            print('MODEL_SAVED')
                            metric_best_score = vl_end_metrics[based_on_metric]


            else:
                if save_model_based_on_loss == True:
                    if train_loss < loss_best_score:
                        save_utils = {'model' : self.state_dict(), 'best_score' : loss_best_score, 'epochs' : i+1}
                        torch.save(save_utils, 'Based-on-loss-' + model_path)
                        print('MODEL_SAVED')
                        loss_best_score = train_loss

                if save_model_based_on_metric == True:
                    if metric_order == 'less_than':
                        if tr_end_metrics[based_on_metric] < metric_best_score:
                            save_utils = {'model' : self.state_dict(), 'best_score' : loss_best_score, 'epochs' : i+1}
                            torch.save(save_utils, model_path)
                            print('MODEL_SAVED')
                            metric_best_score = tr_end_metrics[based_on_metric]

                    elif metric_order == 'greater_than':
                        if tr_end_metrics[based_on_metric] > metric_best_score:
                            save_utils = {'model' : self.state_dict(), 'best_score' : loss_best_score, 'epochs' : i+1}
                            torch.save(save_utils, model_path)
                            print('MODEL_SAVED')
                            metric_best_score = tr_end_metrics[based_on_metric]



            if validset:
                print('train metrics : {}'.format(temp_tr_metrics))
                print('valid metrics : {}'.format(temp_vl_metrics))
                print("Epoch : {} train_loss : {} valid_loss : {}".format(i+1,train_loss,valid_loss))
            else:
                print('train metrics : {}'.format(temp_tr_metrics))
                print("Epoch : {} train_loss : {} ".format(i+1,train_loss))
                
