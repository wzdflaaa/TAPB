import sys
import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve,accuracy_score,f1_score
from prettytable import PrettyTable
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, output_path, config):
        self.model = model
        self.optim = optim
        self.device = device
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.epochs = config.TRAIN.MAX_EPOCH

        self.current_epoch = 0
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.step = 0

        #self.nb_training = len(self.train_dataloader)

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        
        #self.train_model_loss_epoch = []
        
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = output_path
        
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC"]
        self.val_table = PrettyTable(valid_metric_header)
        
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold"]
        self.test_table = PrettyTable(test_metric_header)
        
        train_metric_header = ["# Epoch", "Train_loss"]
        self.train_table = PrettyTable(train_metric_header)

        
        #训练使用默认的反事实分支系数
        self.train_lambda_drug=float(config.TRAIN.LAMBDA_DRUG)
        self.train_lambda_protein=float(config.TRAIN.LAMBDA_PROTEIN)
        #评估时最终使用系数__网格搜索覆盖
        self.eval_lambda_drug_grid =self.train_lambda_drug
        self.eval_lambda_protein_grid = self.train_lambda_protein
        #网格搜索
        self.enable_lambda_grid_search=bool(config.TRAIN.ENABLE_LAMBDA_GRID_SEARCH)
        self.lambda_drug_grid=list(config.TRAIN.LAMBDA_DRUG_GRID)
        self.lambda_protein_grid=list(config.TRAIN.LAMBDA_PROTEIN_GRID)
        
        
    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            
            self.train_loss_epoch.append(train_loss)
            
            #auroc, auprc = self.test(dataloader="val")
            #验证集评估 -使用默认lambda
            auroc, auprc = self.test(dataloader="val",model=self.model, lambda_drug=self.train_lambda_drug, lambda_protein=self.train_lambda_protein)
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc]))
            self.val_table.add_row(val_lst)
            
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print(' Validation at Epoch ' + str(self.current_epoch), "with AUROC "+ str(auroc) + " AUPRC " + str(auprc))
        
        #最终测试前 对beat_model 在验证集上做网格搜索lambda
        self._select_best_lambdas()

        test_out = self.test(
            dataloader="test",
            model_ref=self.best_model,
            lambda_drug=self.eval_lambda_drug,
            lambda_protein=self.eval_lambda_protein,
        )
        
        auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim = test_out[:7]        
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim]))
        self.test_table.add_row(test_lst)
       
        self.test_metrics.update({
            "auroc": auroc,
            "auprc": auprc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
            "thred_optim": thred_optim,
            "best_epoch": self.best_epoch,
            "F1": f1,
            "best_lambda_drug": self.eval_lambda_drug_grid,
            "best_lambda_protein": self.eval_lambda_protein_grid
        })
        self.test_metrics.update(test_out[7])
        
        #模型摘要输出--保留
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + " with AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        
        self.save_result()

        return self.test_metrics, self.best_epoch
    
    #网格搜索
    def _select_best_lambdas(self):
        #若关闭网格搜索 沿用训练默认值
        if not self.enable_lambda_grid_search:
            return
        best_auroc=-1.0
        best_pair=(self.train_lambda_drug,self.train_lambda_protein)
        #穷举搜索 以验证集AUROC来选
        for ld in self.lambda_drug_grid:
            for lp in self.lambda_protein_grid:
                auroc, auprc = self.test(dataloader="val",model=self.best_model, lambda_drug=float(ld), lambda_protein=float(lp))
                if np.isnan(auroc):
                    continue
                if auroc>best_auroc:
                    best_auroc=auroc
                    best_pair=(float(ld),float(lp))
        self.eval_lambda_drug_grid = best_pair[0]
        self.eval_lambda_protein_grid = best_pair[1]
        print(f"[Lambda Grid Search] best_lambda_drug={self.eval_lambda_drug}, best_lambda_protein={self.eval_lambda_protein}, val_auroc={best_auroc:.6f}")
        return 
    
    def save_result(self):
        if self.config.TRAIN.SAVE_MODEL:
            torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, f"best_epoch_{self.best_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))
        if self.config.TRAIN.SAVE_LAST_EPOCH:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"last_epoch.pth"))
        
        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
       
        with open(os.path.join(self.output_dir,"valid_markdowntable.txt"),'w') as fp:  
            fp.write(self.val_table.get_string())
        with open(os.path.join(self.output_dir, "test_markdowntable.txt"), 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(os.path.join(self.output_dir, "train_markdowntable.txt"), 'w') as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        r=0
        num_batches = len(self.train_dataloader)
        loop = tqdm(self.train_dataloader, colour='#ff4777', file=sys.stdout)
        loop.set_description(f'Train Epoch[{self.current_epoch}/{self.epochs}]')
        for step, batch in enumerate(loop):
            self.optim.zero_grad()
            self.step += 1
            r += 1
            
            input_drugs = {k:v.to(self.device) for k,v in batch['batch_inputs_drug'].items()}
            #input_drugs = batch['batch_inputs_drug'].to(self.device)
            input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
            pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
            labels = torch.tensor(batch['labels']).to(self.device)
            drug_labels = batch['masked_drug_labels']
            
            kwargs={
                'pr_mask': pr_mask,
                #训练阶段三个分支同时走 训练使用默认lamnda
                'lambda_drug': self.train_lambda_drug,
                'lambda_protein': self.train_lambda_protein,
            }
            
            
            if drug_labels is not None:
                #inputs_drugs_m = batch['batch_inputs_drug_m'].to(self.device)
                inputs_drugs_m = {k:v.to(self.device) for k,v in batch['batch_inputs_drug_m'].items()}
                drug_labels = drug_labels.to(self.device)
                
                output = self.model(input_drugs, input_proteins, pr_mask=pr_mask, masked_drugs=inputs_drugs_m,**kwargs)
                
                """b_loss = cross_entropy(output['logits'], labels)
                mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)(output['drug_mlm_logits'], drug_labels)
                loss = b_loss + mlm_loss"""
                
            else:
                output = self.model(input_drugs, input_proteins, pr_mask=pr_mask)
                #loss = cross_entropy(output['logits'], labels)

            #损失规划
            #factual 分支 CE (没有权重)
            factual_loss = cross_entropy(output['factual_logits'], labels)
            #debiased 分支 CE (可学习权重)
            debias_ce_weight = self.model.get_debias_ce_weight() 
            debias_loss = cross_entropy(output['debias_logits'], labels) 
            
            loss = factual_loss + debias_ce_weight * debias_loss
            
            if drug_labels is not None:
                mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)(output['drug_mlm_logits'], drug_labels)
                loss = loss + mlm_loss
            
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            loop.set_postfix(avg_loss=loss_epoch / r,debias_w=float(debias_ce_weight.detach().cpu().numpy()))

        loss_epoch = loss_epoch / num_batches
        return loss_epoch

    def _safe_auc(self,y_true,y_score):
        if len(set(y_true))<2:
            return float('nan'),float('nan')
        return roc_auc_score(y_true,y_score),average_precision_score(y_true,y_score)
    def _group_prior_correlation(self,y_true,y_pred,group_ids):
        stats={}
        for g,y,p in zip(group_ids,y_true,y_pred,group_ids):
            if g not in stats:
                stats[g]={'y':[],'p':[]}
            stats[g]['y'].append(y)
            stats[g]['y'].append(p)
        true_rates=[np.mean(v['y'] for v in stats.values())]
        pred_means=[np.mean(v['p'] for v in stats.values())]
        
        if len(true_rates)<2 or np.std(true_rates)==0 or np.std(pred_means==0):
            return float('nan')
        return float(np.corrcoef(true_rates,pred_means)[0,1])
                    
    
    def test(self, dataloader="test",model_ref=None,lambda_drug=None,lambda_protein=None):
        
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        
        model_ref=self.best_model if model_ref is None and dataloader=="test"else(self.model if model_ref is None else model_ref)
        lambda_drug=self.eval_lambda_drug if lambda_drug is None else lambda_drug
        lambda_protein=self.eval_lambda_protein if lambda_protein is None else lambda_protein
        
        y_label= []
        pr_ids,smiles_ids=[],[]
        preds={'debiased':[],'factual':[],'cf_drug':[],'cf_protein':[]}
        
        loop = tqdm(data_loader, colour='#f47983', file=sys.stdout)
        
        with torch.no_grad():
            model_ref.eval()
            if dataloader == "val":
                loop.set_description(f'Validation')
            elif dataloader == "test":
                loop.set_description(f'Test')
                
            for step, batch in enumerate(loop):
                labels = batch['labels']
                input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
                input_drugs = batch['batch_inputs_drug'].to(self.device)
                pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
                
                """if dataloader == "val":
                    output = self.model(input_drugs, input_proteins, pr_mask=pr_mask)
                elif dataloader == "test":
                    output = self.best_model(input_drugs, input_proteins, pr_mask=pr_mask)"""
                
                output=model_ref(
                    input_drugs,
                    input_proteins,
                    pr_mask=pr_mask,
                    lambda_drug=lambda_drug,
                    lambda_protein=lambda_protein
                )
                
                y_label.extend(labels)
                pr_ids.extend(batch['pr_ids'])
                smiles_ids.extend(batch['smiles'])
                preds['debiased'].extend(output['debiased_logits'][:, 1].tolist())
                preds['factual'].extend(output['factual_logits'][:, 1].tolist())
                preds['cf_drug'].extend(output['cf_drug_logits'][:, 1].tolist())
                preds['cf_protein'].extend(output['cf_protein_logits'][:,1].tolist())
                
        auroc,auprc=self._safe_suc(y_label,preds['debiased'])
        
        
        if dataloader == "val":
            return auroc, auprc
        fpr,tpr,thresholds=roc_curve(y_label,preds['debiased'])
        optimal_idx=np.argmax(tpr-fpr)
        optimal_threshold=thresholds[optimal_idx]
        y_pred_bin=(np.array(preds['debiased']) >= optimal_threshold).astype(int)
        tn,fp,fn,tp=confusion_matrix(y_label,y_pred_bin).ravel()
        
        #去偏分析相关指标
        factual_auc = self._safe_auc(y_label, preds['factual'])
        cf_drug_auc = self._safe_auc(y_label, preds['cf_drug'])
        cf_protein_auc = self._safe_auc(y_label, preds['cf_protein'])
        extra = {
            'factual_auroc': factual_auc[0],
            'factual_auprc': factual_auc[1],
            'cf_drug_auroc': cf_drug_auc[0],
            'cf_drug_auprc': cf_drug_auc[1],
            'cf_protein_auroc': cf_protein_auc[0],
            'cf_protein_auprc': cf_protein_auc[1],
            'debias_effect_drug_auroc_gap': auroc - cf_drug_auc[0],
            'debias_effect_protein_auroc_gap': auroc - cf_protein_auc[0],
            'factual_pos_prob_mean': float(np.mean(preds['factual'])),
            'factual_pos_prob_var': float(np.var(preds['factual'])),
            'debiased_pos_prob_mean': float(np.mean(preds['debiased'])),
            'debiased_pos_prob_var': float(np.var(preds['debiased'])),
            'cf_drug_pos_prob_mean': float(np.mean(preds['cf_drug'])),
            'cf_drug_pos_prob_var': float(np.var(preds['cf_drug'])),
            'cf_protein_pos_prob_mean': float(np.mean(preds['cf_protein'])),
            'cf_protein_pos_prob_var': float(np.var(preds['cf_protein'])),
            'drug_prior_pred_corr': self._group_prior_correlation(y_label, preds['debiased'], smiles_ids),
            'protein_prior_pred_corr': self._group_prior_correlation(y_label, preds['debiased'], pr_ids),
            'learned_debias_ce_weight': float(model_ref.get_debias_ce_weight().detach().cpu()),
        }
        acc = accuracy_score(y_label, y_pred_bin)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = f1_score(y_label, y_pred_bin)
        return auroc, auprc, f1, sensitivity, specificity, acc, optimal_threshold, extra
def binary_cross_entropy(n, labels):
    loss_fct = torch.nn.BCELoss()
    loss = loss_fct(n, labels.float())
    return loss

def cross_entropy(preds, targets, reduction='none'):
    preds = torch.log(preds.clamp_min(1e-2)) #辅助性修改
    loss_f = nn.NLLLoss()
    loss = loss_f(preds, targets.long())
    return loss

