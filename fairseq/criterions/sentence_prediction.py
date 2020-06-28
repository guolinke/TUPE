# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and \
               'sentence_classification_head' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.args.regression_target:
            loss = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                reduction='sum',
            )
            if self.args.num_classes == 2:
                tp = ((logits[:, 0] <= logits[:, 1]) & (targets == 1)).long().sum()
                fp = ((logits[:, 0] <= logits[:, 1]) & (targets == 0)).long().sum()
                fn = ((logits[:, 0] > logits[:, 1]) & (targets == 1)).long().sum()
                tn = ((logits[:, 0] > logits[:, 1]) & (targets == 0)).long().sum()
                assert (tp + fp + tn + fn) == targets.size(0), 'invalid size'
        else:
            logits = logits.squeeze().float()
            targets = targets.float()
            loss = F.mse_loss(
                logits,
                targets,
                reduction='sum',
            )

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        if not self.args.regression_target:
            preds = logits.max(dim=1)[1]
            logging_output.update(
                ncorrect=(preds == targets).sum().item()
            )
            if self.args.num_classes == 2:
                logging_output.update(tp=utils.item(tp.data) if reduce else tp.data)
                logging_output.update(fp=utils.item(fp.data) if reduce else fp.data)
                logging_output.update(fn=utils.item(fn.data) if reduce else fn.data)
                logging_output.update(tn=utils.item(tn.data) if reduce else tn.data)
        else:
            logging_output.update(x=logits.detach().cpu().numpy())
            logging_output.update(y=targets.detach().cpu().numpy())
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect / nsentences)

            tp_sum = sum(log.get('tp', 0) for log in logging_outputs)
            fp_sum = sum(log.get('fp', 0) for log in logging_outputs)
            fn_sum = sum(log.get('fn', 0) for log in logging_outputs)
            tn_sum = sum(log.get('tn', 0) for log in logging_outputs)
            if tp_sum + fp_sum + fn_sum + tn_sum > 0:
                assert tp_sum + fp_sum + fn_sum + tn_sum == sample_size, 'invalid size when aggregating'
                acc = (tp_sum + tn_sum) / sample_size
                tmp = 2 * tp_sum + fp_sum + fn_sum
                f1 = (2 * tp_sum) / tmp if tmp else 0
                tmp = (tp_sum + fp_sum) * (tp_sum + fn_sum) * (tn_sum + fp_sum) * (tn_sum + fn_sum)
                mcc = (tp_sum * tn_sum - fp_sum * fn_sum) / (tmp ** 0.5) if tmp else 0
                agg_output.update(f1=f1)
                agg_output.update(mcc=mcc)
                agg_output.update(acc_f1=0.5 * (acc + f1))

        if len(logging_outputs) > 0 and 'x' in logging_outputs[0]:
            x = np.concatenate([log.get('x', np.array([])) for log in logging_outputs])
            y = np.concatenate([log.get('y', np.array([])) for log in logging_outputs])
            pearson = stats.pearsonr(x, y)[0]
            spearman = stats.spearmanr(x, y)[0]
            agg_output.update(pearson=pearson)
            agg_output.update(spearman=spearman)
            agg_output.update(pearson_spearman=0.5 * (pearson + spearman))

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
