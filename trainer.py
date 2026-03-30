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
        self.test_predictions={}
        self.config = config
        self.output_dir = output_path
        
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC"]
        self.val_table = PrettyTable(valid_metric_header)
        
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy","Threshold"]
        self.test_table = PrettyTable(test_metric_header)
        
        train_metric_header = ["# Epoch", "Train_loss"]
        self.train_table = PrettyTable(train_metric_header)

        
        #训练使用默认的反事实分支系数 仅推理/搜索使用 不参与训练loss
        self.train_lambda_drug=float(config.TRAIN.LAMBDA_DRUG)
        self.train_lambda_protein=float(config.TRAIN.LAMBDA_PROTEIN)
        
        #评估时最终使用系数__网格搜索覆盖
        self.eval_lambda_drug =self.train_lambda_drug
        self.eval_lambda_protein = self.train_lambda_protein
        
        self.eval_lambda_drug_only=self.train_lambda_drug
        self.eval_lambda_protein_only=self.train_lambda_protein
        
        #分支loss权重
        self.cf_drug_loss_weight = float(getattr(config.TRAIN,"CF_DRUG_LOSS_WEIGHT",0.3))
        self.cf_protein_loss_weight= float(getattr(config.TRAIN,"CF_PROTEIN_LOSS_WEIGHT",0.3))
        self.mlm_loss_weight=float(getattr(config.TRAIN,"MLM_LOSS_WEIGHT",0.3))
        
        #最佳epoch的选择依据：factual分支性能
        self.select_best_epoch_by=str(getattr(config.TRAIN,"SELECT_BEST_EPOCH_BY","factual")).lower()
        
        #网格搜索
        self.enable_lambda_grid_search=bool(config.TRAIN.ENABLE_LAMBDA_GRID_SEARCH)
        #单边去偏独立网格搜索
        #self.lambda_drug_grid=list(config.TRAIN.LAMBDA_DRUG_GRID)
        #self.lambda_protein_grid=list(config.TRAIN.LAMBDA_PROTEIN_GRID)
        
        #两步搜索测试 ===== coarse-to-fine lambda search =====
        # 默认先用 [-1, 1] 做粗搜索；
        self.lambda_coarse_min = float(getattr(config.TRAIN, "LAMBDA_COARSE_MIN", -1.0))
        self.lambda_coarse_max = float(getattr(config.TRAIN, "LAMBDA_COARSE_MAX", 1.0))
        self.lambda_coarse_step = float(getattr(config.TRAIN, "LAMBDA_COARSE_STEP", 0.5))
        self.lambda_fine_radius = float(getattr(config.TRAIN, "LAMBDA_FINE_RADIUS", 0.5))
        self.lambda_fine_step = float(getattr(config.TRAIN, "LAMBDA_FINE_STEP", 0.1))
        
    def train(self):
        float2str = lambda x: '%0.4f' % x

        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()

            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            # 这里不再用 debiased@默认lambda 选最佳epoch # 改为直接看 factual 分支的验证集表现
            val_metrics = self.test(
                dataloader="val",
                model_ref=self.model,
                lambda_drug=0.0,
                lambda_protein=0.0,
                return_branch_metrics=True,
            )

            factual_auroc = val_metrics["auroc"]["factual"]
            factual_auprc = val_metrics["auprc"]["factual"]

            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [factual_auroc, factual_auprc]))
            self.val_table.add_row(val_lst)
            self.val_auroc_epoch.append(factual_auroc)

            if factual_auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = factual_auroc
                self.best_epoch = self.current_epoch

            print(
                f" Validation at Epoch {self.current_epoch} "
                f"with Factual AUROC {factual_auroc} AUPRC {factual_auprc}"
            )

        # 训练完成后，再在 best_model 上做 coarse-to-fine lambda 搜索
        self._select_best_lambdas()

        test_out = self.test(
            dataloader="test",
            model_ref=self.best_model,
            lambda_drug=self.eval_lambda_drug,
            lambda_protein=self.eval_lambda_protein,
        )

        auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim = test_out[:7]
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [
            auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim
        ]))
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
            "best_lambda_drug": self.eval_lambda_drug,
            "best_lambda_protein": self.eval_lambda_protein,
        })

        print(
            'Test at Best Model of Epoch ' + str(self.best_epoch) + " with AUROC "
            + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity)
            + " Specificity " + str(specificity) + " Accuracy " + str(accuracy)
            + " Thred_optim " + str(thred_optim)
        )

        self.save_result()
        return self.test_metrics, self.best_epoch
    
    def _build_search_values(self, start, end, step):
        vals = []
        cur = float(start)
        end = float(end)
        step = float(step)
        while cur <= end + 1e-8:
            vals.append(round(cur, 10))
            cur += step
        return vals

    def _build_local_values(self, center, radius, global_min, global_max, step):
        start = max(global_min, center - radius)
        end = min(global_max, center + radius)
        return self._build_search_values(start, end, step)

    def _eval_branch_metrics(self, lambda_drug, lambda_protein):
        return self.test(
            dataloader="val",
            model_ref=self.best_model,
            lambda_drug=float(lambda_drug),
            lambda_protein=float(lambda_protein),
            return_branch_metrics=True,
        )
    def _search_joint_grid(self, lambda_drug_values, lambda_protein_values, tag="joint"):
        best_auroc = -1.0
        best_pair = (self.train_lambda_drug, self.train_lambda_protein)
        records = []

        for ld in lambda_drug_values:
            for lp in lambda_protein_values:
                branch_metrics = self._eval_branch_metrics(ld, lp)
                auroc = branch_metrics["auroc"].get("debiased", float("nan"))
                auprc = branch_metrics["auprc"].get("debiased", float("nan"))

                records.append({
                    "tag": tag,
                    "lambda_drug": float(ld),
                    "lambda_protein": float(lp),
                    "auroc": float(auroc) if not np.isnan(auroc) else float("nan"),
                    "auprc": float(auprc) if not np.isnan(auprc) else float("nan"),
                })

                print(
                    f"[{tag}] lambda_drug={ld:.2f}, lambda_protein={lp:.2f}, "
                    f"val_auroc={auroc:.6f}, val_auprc={auprc:.6f}"
                )

                if np.isnan(auroc):
                    continue
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_pair = (float(ld), float(lp))

        return best_pair, best_auroc, records
    def _search_single_grid(self, branch_name, candidate_values, fixed_other_lambda=0.0, tag="single"):
        best_auroc = -1.0
        best_lambda = None
        records = []

        for val in candidate_values:
            if branch_name == "drug_only":
                branch_metrics = self._eval_branch_metrics(val, fixed_other_lambda)
                metric_key = "debiased_drug_only"
            elif branch_name == "protein_only":
                branch_metrics = self._eval_branch_metrics(fixed_other_lambda, val)
                metric_key = "debiased_protein_only"
            else:
                raise ValueError(f"Unsupported branch_name: {branch_name}")

            auroc = branch_metrics["auroc"].get(metric_key, float("nan"))
            auprc = branch_metrics["auprc"].get(metric_key, float("nan"))

            records.append({
                "tag": tag,
                "branch": branch_name,
                "lambda": float(val),
                "auroc": float(auroc) if not np.isnan(auroc) else float("nan"),
                "auprc": float(auprc) if not np.isnan(auprc) else float("nan"),
            })

            print(f"[{tag}-{branch_name}] lambda={val:.2f}, "
                    f"val_auroc={auroc:.6f}, val_auprc={auprc:.6f}")

            if np.isnan(auroc):
                continue
            if auroc > best_auroc:
                best_auroc = auroc
                best_lambda = float(val)

        if best_lambda is None:
            best_lambda = self.train_lambda_drug if branch_name == "drug_only" else self.train_lambda_protein

        return best_lambda, best_auroc, records
    #网格搜索
    def _build_search_values(self, start, end, step):
        vals = []
        cur = float(start)
        end = float(end)
        step = float(step)
        while cur <= end + 1e-8:
            vals.append(round(cur, 10))
            cur += step
        return vals
    
    def _build_local_values(self, center, radius, global_min, global_max, step):
        start = max(global_min, center - radius)
        end = min(global_max, center + radius)
        return self._build_search_values(start, end, step)


    def _eval_branch_metrics(self, lambda_drug, lambda_protein):
        return self.test(
            dataloader="val",
            model_ref=self.best_model,
            lambda_drug=float(lambda_drug),
            lambda_protein=float(lambda_protein),
            return_branch_metrics=True,
        )
    
    def _search_single_grid(self, branch_name, candidate_values, fixed_other_lambda=0.0, tag="single"):
        best_auroc = -1.0
        best_lambda = None
        records = []

        for val in candidate_values:
            if branch_name == "drug_only":
                branch_metrics = self._eval_branch_metrics(val, fixed_other_lambda)
                metric_key = "debiased_drug_only"
            elif branch_name == "protein_only":
                branch_metrics = self._eval_branch_metrics(fixed_other_lambda, val)
                metric_key = "debiased_protein_only"
            else:
                raise ValueError(f"Unsupported branch_name: {branch_name}")

            auroc = branch_metrics["auroc"].get(metric_key, float("nan"))
            auprc = branch_metrics["auprc"].get(metric_key, float("nan"))

            records.append({
                "tag": tag,
                "branch": branch_name,
                "lambda": float(val),
                "auroc": float(auroc) if not np.isnan(auroc) else float("nan"),
                "auprc": float(auprc) if not np.isnan(auprc) else float("nan"),
            })

            print(f"[{tag}-{branch_name}] lambda={val:.2f}, val_auroc={auroc:.6f}, val_auprc={auprc:.6f}")

            if np.isnan(auroc):
                continue
            if auroc > best_auroc:
                best_auroc = auroc
                best_lambda = float(val)

        if best_lambda is None:
            best_lambda = 0.0

        return best_lambda, best_auroc, records


    def _select_best_lambdas(self):
        if not self.enable_lambda_grid_search:
            return

        coarse_min = self.lambda_coarse_min
        coarse_max = self.lambda_coarse_max
        coarse_step = self.lambda_coarse_step
        fine_radius = self.lambda_fine_radius
        fine_step = self.lambda_fine_step

        all_records = {
            "joint_coarse": [],
            "joint_fine": [],
            "drug_only_coarse": [],
            "drug_only_fine": [],
            "protein_only_coarse": [],
            "protein_only_fine": [],
        }

        # 1) joint coarse
        coarse_ld_values = self._build_search_values(coarse_min, coarse_max, coarse_step)
        coarse_lp_values = self._build_search_values(coarse_min, coarse_max, coarse_step)

        best_pair_coarse, best_joint_auroc_coarse, joint_coarse_records = self._search_joint_grid(
            coarse_ld_values,
            coarse_lp_values,
            tag="joint-coarse"
        )
        all_records["joint_coarse"] = joint_coarse_records

        # 2) joint fine
        fine_ld_values = self._build_local_values(
            center=best_pair_coarse[0],
            radius=fine_radius,
            global_min=coarse_min,
            global_max=coarse_max,
            step=fine_step
        )
        fine_lp_values = self._build_local_values(
            center=best_pair_coarse[1],
            radius=fine_radius,
            global_min=coarse_min,
            global_max=coarse_max,
            step=fine_step
        )

        best_pair_fine, best_joint_auroc_fine, joint_fine_records = self._search_joint_grid(
            fine_ld_values,
            fine_lp_values,
            tag="joint-fine"
        )
        all_records["joint_fine"] = joint_fine_records

        self.eval_lambda_drug = best_pair_fine[0]
        self.eval_lambda_protein = best_pair_fine[1]

        # 3) drug-only
        best_drug_coarse, best_drug_auroc_coarse, drug_coarse_records = self._search_single_grid(
            branch_name="drug_only",
            candidate_values=coarse_ld_values,
            fixed_other_lambda=0.0,
            tag="drug-only-coarse"
        )
        all_records["drug_only_coarse"] = drug_coarse_records

        fine_drug_values = self._build_local_values(
            center=best_drug_coarse,
            radius=fine_radius,
            global_min=coarse_min,
            global_max=coarse_max,
            step=fine_step
        )

        best_drug_fine, best_drug_auroc_fine, drug_fine_records = self._search_single_grid(
            branch_name="drug_only",
            candidate_values=fine_drug_values,
            fixed_other_lambda=0.0,
            tag="drug-only-fine"
        )
        all_records["drug_only_fine"] = drug_fine_records
        self.eval_lambda_drug_only = best_drug_fine

        # 4) protein-only
        best_protein_coarse, best_protein_auroc_coarse, protein_coarse_records = self._search_single_grid(
            branch_name="protein_only",
            candidate_values=coarse_lp_values,
            fixed_other_lambda=0.0,
            tag="protein-only-coarse"
        )
        all_records["protein_only_coarse"] = protein_coarse_records

        fine_protein_values = self._build_local_values(
            center=best_protein_coarse,
            radius=fine_radius,
            global_min=coarse_min,
            global_max=coarse_max,
            step=fine_step
        )

        best_protein_fine, best_protein_auroc_fine, protein_fine_records = self._search_single_grid(
            branch_name="protein_only",
            candidate_values=fine_protein_values,
            fixed_other_lambda=0.0,
            tag="protein-only-fine"
        )
        all_records["protein_only_fine"] = protein_fine_records
        self.eval_lambda_protein_only = best_protein_fine

        torch.save(all_records, os.path.join(self.output_dir, "lambda_search_records.pt"))

        print(
            f"[Lambda Coarse-to-Fine Search] "
            f"joint(best_lambda_drug={self.eval_lambda_drug}, "
            f"best_lambda_protein={self.eval_lambda_protein}, "
            f"val_auroc={best_joint_auroc_fine:.6f}); "
            f"drug_only(best_lambda_drug_only={self.eval_lambda_drug_only}, "
            f"val_auroc={best_drug_auroc_fine:.6f}); "
            f"protein_only(best_lambda_protein_only={self.eval_lambda_protein_only}, "
            f"val_auroc={best_protein_auroc_fine:.6f})"
        )
    
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
        if self.test_predictions:
            torch.save(self.test_predictions,os.path.join(self.output_dir,"test_precdictions.pt"))
        if self.config.TRAIN.SAVE_LAST_EPOCH:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"last_epoch.pth"))
        
        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")   
        with open(os.path.join(self.output_dir,"valid_markdowntable.txt"),'w',encoding='utf-8') as fp:  
            fp.write(self.val_table.get_string())
        with open(os.path.join(self.output_dir, "test_markdowntable.txt"), 'w',encoding='utf-8') as fp:
            fp.write(self.test_table.get_string())
        with open(os.path.join(self.output_dir, "train_markdowntable.txt"), 'w',encoding='utf-8') as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        r = 0
        num_batches = len(self.train_dataloader)

        loop = tqdm(self.train_dataloader, colour='#ff4777', file=sys.stdout)
        loop.set_description(f'Train Epoch[{self.current_epoch}/{self.epochs}]')

        for step, batch in enumerate(loop):
            self.optim.zero_grad()
            self.step += 1
            r += 1

            input_drugs = {k: v.to(self.device) for k, v in batch['batch_inputs_drug'].items()}
            input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
            pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
            labels = torch.tensor(batch['labels']).to(self.device)
            drug_labels = batch['masked_drug_labels']

            kwargs = {
                'lambda_drug': self.train_lambda_drug,
                'lambda_protein': self.train_lambda_protein,
            }

            if drug_labels is not None:
                inputs_drugs_m = {k: v.to(self.device) for k, v in batch['batch_inputs_drug_m'].items()}
                drug_labels = drug_labels.to(self.device)
                output = self.model(
                    input_drugs,
                    input_proteins,
                    pr_mask=pr_mask,
                    masked_drugs=inputs_drugs_m,
                    **kwargs
                )
            else:
                output = self.model(input_drugs, input_proteins, pr_mask=pr_mask, **kwargs)

            # ===== 新训练机制 loss =====
            factual_loss = cross_entropy(output['factual_logits'], labels)
            cf_drug_loss = cross_entropy(output['cf_drug_logits'], labels)
            cf_protein_loss = cross_entropy(output['cf_protein_logits'], labels)

            loss = (
                factual_loss
                + self.cf_drug_loss_weight * cf_drug_loss
                + self.cf_protein_loss_weight * cf_protein_loss
            )

            mlm_loss_value = None
            if drug_labels is not None and output['drug_mlm_logits'] is not None:
                mlm_loss_value = nn.CrossEntropyLoss(ignore_index=-1)(output['drug_mlm_logits'],drug_labels)
                loss = loss + self.mlm_loss_weight * mlm_loss_value

            loss.backward()
            self.optim.step()

            loss_epoch += loss.item()

            postfix_dict = {
                "avg_loss": loss_epoch / r,
                "Lf": float(factual_loss.detach().cpu().item()),
                "Lcf_d": float(cf_drug_loss.detach().cpu().item()),
                "Lcf_t": float(cf_protein_loss.detach().cpu().item()),
            }
            if mlm_loss_value is not None:
                postfix_dict["Lmlm"] = float(mlm_loss_value.detach().cpu().item())

            loop.set_postfix(**postfix_dict)

        loss_epoch = loss_epoch / num_batches
        return loss_epoch

    def _safe_auc(self,y_true,y_score):
        if len(set(y_true))<2:
            return float('nan'),float('nan')
        return roc_auc_score(y_true,y_score),average_precision_score(y_true,y_score)
    def _group_prior_correlation(self,y_true,y_pred,group_ids):
        stats = {}
        for g, y, p in zip(group_ids,y_true,y_pred):
            if g not in stats:
                stats[g]={'y':[],'p':[]}
            stats[g]['y'].append(y)
            stats[g]['p'].append(p)
        true_rates=[np.mean(v['y']) for v in stats.values()]
        pred_means=[np.mean(v['p']) for v in stats.values()]
        
        if len(true_rates)<2 or np.std(true_rates)==0 or np.std(pred_means)==0 :
            return float('nan')
        return float(np.corrcoef(true_rates,pred_means)[0,1])
                    
    
    def test(self, dataloader="test",model_ref=None,lambda_drug=None,lambda_protein=None,return_branch_metrics=False):
        
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
        preds={'debiased':[],'factual':[],'cf_drug':[],'cf_protein':[],'debiased_drug_only': [],
            'debiased_protein_only': [],}
        logits={'debiased_logits':[],'factual_logits':[],'cf_drug_logits':[],'cf_protein_logits':[],'debiased_drug_only_logits': [],
            'debiased_protein_only_logits': [],}
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
                #input_drugs = batch['batch_inputs_drug'].to(self.device)
                input_drugs={k: v.to(self.device) for k,v in batch['batch_inputs_drug'].items()}
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
                preds['debiased'].extend(output['prob']['debiased'][:, 1].tolist())
                preds['factual'].extend(output['prob']['factual'][:, 1].tolist())
                preds['cf_drug'].extend(output['prob']['cf_drug'][:, 1].tolist())
                preds['cf_protein'].extend(output['prob']['cf_protein'][:,1].tolist())
                preds['debiased_drug_only'].extend(output['prob']['debiased_drug_only'][:, 1].tolist())
                preds['debiased_protein_only'].extend(output['prob']['debiased_protein_only'][:, 1].tolist())
                logits['debiased_logits'].extend(output['debiased_logits'].tolist())
                logits['factual_logits'].extend(output['factual_logits'].tolist())
                logits['cf_drug_logits'].extend(output['cf_drug_logits'].tolist())
                logits['cf_protein_logits'].extend(output['cf_protein_logits'].tolist())
                logits['debiased_drug_only_logits'].extend(output['debiased_drug_only_logits'].tolist())
                logits['debiased_protein_only_logits'].extend(output['debiased_protein_only_logits'].tolist())
        
        branch_auc = {k: self._safe_auc(y_label, v) for k, v in preds.items()}
        auroc, auprc = branch_auc['debiased']

        if dataloader == "val" and return_branch_metrics:
            return {
                'auroc': {k: float(v[0]) for k, v in branch_auc.items()},
                'auprc': {k: float(v[1]) for k, v in branch_auc.items()},
            }
        if dataloader == "val":
            return auroc, auprc
    
        fpr,tpr,thresholds=roc_curve(y_label,preds['debiased'])
        optimal_idx=np.argmax(tpr-fpr)
        optimal_threshold=thresholds[optimal_idx]
        y_pred_bin=(np.array(preds['debiased']) >= optimal_threshold).astype(int)
        tn,fp,fn,tp=confusion_matrix(y_label,y_pred_bin).ravel()
        
        self.test_predictions={
            'label': list(map(int, y_label)),
            'factual_logits': logits['factual_logits'],
            'cf_drug_logits': logits['cf_drug_logits'],
            'cf_protein_logits': logits['cf_protein_logits'],
            'debiased_drug_only_logits': logits['debiased_drug_only_logits'],
            'debiased_protein_only_logits': logits['debiased_protein_only_logits'],
            'debiased_logits': logits['debiased_logits'],
            'pr_id': pr_ids,
            'SMILES': smiles_ids,
        }
        acc = accuracy_score(y_label, y_pred_bin)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = f1_score(y_label, y_pred_bin)
        return auroc, auprc, f1, sensitivity, specificity, acc, optimal_threshold

def binary_cross_entropy(n, labels):
    loss_fct = torch.nn.BCELoss()
    loss = loss_fct(n, labels.float())
    return loss

def cross_entropy(preds, targets, reduction='mean'):
    #preds: shape [B], raw logitstargets: shape [B], values in {0,1}
    targets = targets.float()
    loss_f = nn.BCEWithLogitsLoss(reduction=reduction)
    loss = loss_f(preds.view(-1), targets.view(-1))
    return loss
