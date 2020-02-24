import os
import json
from logging import getLogger
from time import perf_counter
import pandas as pd
import torch
from ignite.engine.engine import Engine, Events
from ignite.metrics import Average
import torch.nn.functional as F


def accuracy(y, t):
    pred_label = torch.argmax(y, dim=1)
    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc

def save_json(filepath, params):
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)


class DictOutputTransform:
    def __init__(self, key, index=0):
        self.key = key
        self.index = index

    def __call__(self, x):
        if self.index >= 0:
            x = x[self.index]
        return x[self.key]


def create_trainer(classifier, optimizer, device):
    classifier.to(device)

    def update_fn(engine, batch):
        classifier.train()
        optimizer.zero_grad()
        # batch = [elem.to(device) for elem in batch]
        x,y = [elem.to(device) for elem in batch]

        x = x.to(device,dtype = torch.float)
        grapheme_root = y[:,0].to(device,dtype = torch.long)
        vowel_diacritic = y[:,1].to(device,dtype = torch.long)
        consonant_diacritic = y[:,2].to(device,dtype =torch.long)

        preds = classifier(x)
        loss_grapheme = F.cross_entropy(preds[0], grapheme_root)
        loss_vowel = F.cross_entropy(preds[1], vowel_diacritic)
        loss_consonant = F.cross_entropy(preds[2], consonant_diacritic)
        loss = loss_grapheme + loss_vowel + loss_consonant

        metrics = {
            'loss': loss.item(),
            'loss_grapheme': loss_grapheme.item(),
            'loss_vowel': loss_vowel.item(),
            'loss_consonant': loss_consonant.item(),
            'acc_grapheme': accuracy(preds[0], grapheme_root),
            'acc_vowel': accuracy(preds[1], vowel_diacritic),
            'acc_consonant': accuracy(preds[2], consonant_diacritic),
        }


        loss.backward()
        optimizer.step()
        return metrics, preds, (grapheme_root,vowel_diacritic,consonant_diacritic)
    trainer = Engine(update_fn)

    for key in classifier.metrics_keys:
        Average(output_transform=DictOutputTransform(key)).attach(trainer, key)
    return trainer


def create_evaluator(classifier, device):
    classifier.to(device)

    def update_fn(engine, batch):
        classifier.eval()
        with torch.no_grad():
            # batch = [elem.to(device) for elem in batch]
            x,y = [elem.to(device) for elem in batch]

            x = x.to(device,dtype = torch.float)
            grapheme_root = y[:,0].to(device,dtype = torch.long)
            vowel_diacritic = y[:,1].to(device,dtype = torch.long)
            consonant_diacritic = y[:,2].to(device,dtype =torch.long)

            preds = classifier(x)
            loss_grapheme = F.cross_entropy(preds[0], grapheme_root)
            loss_vowel = F.cross_entropy(preds[1], vowel_diacritic)
            loss_consonant = F.cross_entropy(preds[2], consonant_diacritic)
            loss = loss_grapheme + loss_vowel + loss_consonant

            metrics = {
                'loss': loss.item(),
                'loss_grapheme': loss_grapheme.item(),
                'loss_vowel': loss_vowel.item(),
                'loss_consonant': loss_consonant.item(),
                'acc_grapheme': accuracy(preds[0], grapheme_root),
                'acc_vowel': accuracy(preds[1], vowel_diacritic),
                'acc_consonant': accuracy(preds[2], consonant_diacritic),
            }
            return metrics, preds, (grapheme_root,vowel_diacritic,consonant_diacritic)
    evaluator = Engine(update_fn)

    for key in classifier.metrics_keys:
        Average(output_transform=DictOutputTransform(key)).attach(evaluator, key)
    return evaluator


class LogReport:
    def __init__(self, evaluator=None, dirpath=None, logger=None):
        self.evaluator = evaluator
        self.dirpath = str(dirpath) if dirpath is not None else None
        self.logger = logger or getLogger(__name__)

        self.reported_dict = {}  # To handle additional parameter to monitor
        self.history = []
        self.start_time = perf_counter()

    def report(self, key, value):
        self.reported_dict[key] = value

    def __call__(self, engine):
        elapsed_time = perf_counter() - self.start_time
        elem = {'epoch': engine.state.epoch,
                'iteration': engine.state.iteration}
        elem.update({f'train/{key}': value
                     for key, value in engine.state.metrics.items()})
        if self.evaluator is not None:
            elem.update({f'valid/{key}': value
                         for key, value in self.evaluator.state.metrics.items()})
        elem.update(self.reported_dict)
        elem['elapsed_time'] = elapsed_time
        self.history.append(elem)
        if self.dirpath:
            save_json(os.path.join(self.dirpath, 'log.json'), self.history)
            self.get_dataframe().to_csv(os.path.join(self.dirpath, 'log.csv'), index=False)

        # --- print ---
        msg = ''
        for key, value in elem.items():
            if key in ['iteration']:
                # skip printing some parameters...
                continue
            elif isinstance(value, int):
                msg += f'{key} {value: >6d} '
            else:
                msg += f'{key} {value: 8f} '
#         self.logger.warning(msg)
        print(msg)

        # --- Reset ---
        self.reported_dict = {}

    def get_dataframe(self):
        df = pd.DataFrame(self.history)
        return df


class SpeedCheckHandler:
    def __init__(self, iteration_interval=10, logger=None):
        self.iteration_interval = iteration_interval
        self.logger = logger or getLogger(__name__)
        self.prev_time = perf_counter()

    def __call__(self, engine: Engine):
        if engine.state.iteration % self.iteration_interval == 0:
            cur_time = perf_counter()
            spd = self.iteration_interval / (cur_time - self.prev_time)
            self.logger.warning(f'{spd} iter/sec')
            # reset
            self.prev_time = cur_time

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)


class ModelSnapshotHandler:
    def __init__(self, model, filepath='model_{count:06}.pt',
                 interval=1, logger=None):
        self.model = model
        self.filepath: str = str(filepath)
        self.interval = interval
        self.logger = logger or getLogger(__name__)
        self.count = 0

    def __call__(self, engine: Engine):
        self.count += 1
        if self.count % self.interval == 0:
            filepath = self.filepath.format(count=self.count)
            torch.save(self.model.state_dict(), filepath)
            # self.logger.warning(f'save model to {filepath}...')