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

        self.nb_training = len(self.train_dataloader)

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = output_path
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold"]

        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)
        
        @staticmethod
        def _build_override_embedding(emb, mode="zero"):
            if mode == "zero":
                return torch.zeros_like(emb)
            if mode == "mean":
                mean_emb = emb.mean(dim=1, keepdim=True)
                return mean_emb.expand_as(emb)
        raise ValueError(f"Unsupported override mode: {mode}")

        @staticmethod
        def _safe_auroc(y_true, y_score):
            if len(set(y_true)) < 2:
                return float("nan")
        return roc_auc_score(y_true, y_score)

    @staticmethod
    def _summary_stats(values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return {}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
        }

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc = self.test(dataloader="val")

            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc]))
            self.val_table.add_row(val_lst)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print(' Validation at Epoch ' + str(self.current_epoch), "with AUROC "+ str(auroc) + " AUPRC " + str(auprc))
        auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim]))
        self.test_table.add_row(test_lst)
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
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
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
            input_drugs = batch['batch_inputs_drug'].to(self.device)
            input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
            pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
            labels = torch.tensor(batch['labels']).to(self.device)
            drug_labels = batch['masked_drug_labels']
            if drug_labels is not None:
                inputs_drugs_m = batch['batch_inputs_drug_m'].to(self.device)
                drug_labels = drug_labels.to(self.device)
                output = self.model(input_drugs, input_proteins, pr_mask=pr_mask, masked_drugs=inputs_drugs_m)
                b_loss = cross_entropy(output['logits'], labels)
                mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)(output['drug_mlm_logits'], drug_labels)
                loss = b_loss + mlm_loss
            else:
                output = self.model(input_drugs, input_proteins, pr_mask=pr_mask)
                loss = cross_entropy(output['logits'], labels)

            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            loop.set_postfix(avg_loss=loss_epoch / r)

        loss_epoch = loss_epoch / num_batches
        return loss_epoch

    def test(self, dataloader="test"):
        y_label, y_pred = [], []
        
        # Bias diagnosis setup
        diag_stage = str(getattr(self.config.TRAIN, "BIAS_DIAG_STAGE", "test")).lower()
        enable_bias_diag = bool(getattr(self.config.TRAIN, "BIAS_DIAG", False)) and (
            (diag_stage == "both" and dataloader in ["val", "test"]) or
            (diag_stage == dataloader)
        )
        lam_t = float(getattr(self.config.TRAIN, "LAMBDA_T", 1.0))
        lam_d = float(getattr(self.config.TRAIN, "LAMBDA_D", 1.0))
        # always collect both modes for robust diagnosis
        diag_modes = ["zero", "mean"]
        diag_store = {
            mode: {"factual": [], "tonly": [], "donly": [], "debias": []}
            for mode in diag_modes
        }
        
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        
        loop = tqdm(data_loader, colour='#f47983', file=sys.stdout)
        with torch.no_grad():
            #self.model.eval()
            if dataloader == "val":
                self.model.eval()
                active_model = self.model
                loop.set_description(f'Validation')
            else:
                self.best_model.eval()
                active_model = self.best_model
                loop.set_description(f'Test')
                
            for step, batch in enumerate(loop):

                labels = batch['labels']
                input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
                input_drugs = batch['batch_inputs_drug'].to(self.device)
                pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
                
                output = active_model(
                    input_drugs,
                    input_proteins,
                    pr_mask=pr_mask,
                    return_embeddings=enable_bias_diag,
                )
                n_factual = output['logits'][:, 1]
                y_label = y_label + labels
                
                # final evaluation should always be factual score (AUROC model selection consistency)
                y_pred = y_pred + n_factual.tolist()

                if enable_bias_diag:
                    for mode in diag_modes:
                        drug_const = self._build_override_embedding(output['drug_emb'], mode=mode)
                        target_const = self._build_override_embedding(output['target_emb'], mode=mode)

                        output_tonly = active_model(
                            input_drugs,
                            input_proteins,
                            pr_mask=pr_mask,
                            override_drug_emb=drug_const,
                        )
                        output_donly = active_model(
                            input_drugs,
                            input_proteins,
                            pr_mask=pr_mask,
                            override_target_emb=target_const,
                        )

                        n_tonly = output_tonly['logits'][:, 1]
                        n_donly = output_donly['logits'][:, 1]
                        n_debias = n_factual - lam_t * n_tonly - lam_d * n_donly

                        diag_store[mode]["factual"] += n_factual.tolist()
                        diag_store[mode]["tonly"] += n_tonly.tolist()
                        diag_store[mode]["donly"] += n_donly.tolist()
                        diag_store[mode]["debias"] += n_debias.tolist()

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            _prec, _recall, _ = precision_recall_curve(y_label, y_pred)

            # Youden index for the optimal threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            y_pred_array = np.array(y_pred)
            y_pred_bin = (y_pred_array >= optimal_threshold).astype(int)

            # confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_label, y_pred_bin).ravel()


            acc = accuracy_score(y_label, y_pred_bin)
            sensitivity = tp / (tp + fn)  # recall
            specificity = tn / (tn + fp)
            f1 = f1_score(y_label, y_pred_bin)
            
            if enable_bias_diag:
                self.test_metrics["bias_diag_stage"] = diag_stage
                self.test_metrics["lambda_t"] = lam_t
                self.test_metrics["lambda_d"] = lam_d
                self.test_metrics["bias_diag"] = {}
                for mode in diag_modes:
                    factual = np.asarray(diag_store[mode]["factual"])
                    tonly = np.asarray(diag_store[mode]["tonly"])
                    donly = np.asarray(diag_store[mode]["donly"])
                    debias = np.asarray(diag_store[mode]["debias"])
                    self.test_metrics["bias_diag"][mode] = {
                        "logit_factual": factual.tolist(),
                        "logit_tonly": tonly.tolist(),
                        "logit_donly": donly.tolist(),
                        "logit_debias": debias.tolist(),
                        "delta_tonly": (factual - tonly).tolist(),
                        "delta_donly": (factual - donly).tolist(),
                        "auroc_factual": self._safe_auroc(y_label, factual),
                        "auroc_tonly": self._safe_auroc(y_label, tonly),
                        "auroc_donly": self._safe_auroc(y_label, donly),
                        "auroc_debias": self._safe_auroc(y_label, debias),
                        "summary_factual": self._summary_stats(factual),
                        "summary_tonly": self._summary_stats(tonly),
                        "summary_donly": self._summary_stats(donly),
                        "summary_debias": self._summary_stats(debias),
                    }

            
            return auroc, auprc, f1, sensitivity, specificity, acc, optimal_threshold
        else:
            return auroc, auprc

def binary_cross_entropy(n, labels):
    loss_fct = torch.nn.BCELoss()
    loss = loss_fct(n, labels.float())
    return loss


def cross_entropy(preds, targets, reduction='none'):
    preds = torch.log(preds)
    loss_f = nn.NLLLoss()
    loss = loss_f(preds, targets.long())
    return loss
