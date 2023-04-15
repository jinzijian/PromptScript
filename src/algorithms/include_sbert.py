import os, sys
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from .include_sbert_evaluator import CustomBinaryClassificationEvaluator
import core
import datasets
import metrics
import nets
import torch
import pandas as pd

   

def train(args, model, train_dataset, val_dataset=None, scheduler=None, optimizer=None, logger=None, metric=None):
    train_dataloader = core.builder.make_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    loss_fn = core.builder.make_loss(tag=args.loss, model=model)
    val_text_0, val_text_1, val_labels = datasets.common.sbert_data_to_lists(val_dataset)
    print(f'train pos ratio={datasets.common.sbert_data_class_ratio(train_dataset):3f}, test pos ratio={datasets.common.sbert_data_class_ratio(val_dataset):3f}')
    # evaluator = BinaryClassificationEvaluator(val_text_0, val_text_1, val_labels, batch_size=args.get('val_batch_size', args.batch_size))
    evaluator = CustomBinaryClassificationEvaluator(val_text_0, val_text_1, val_labels, batch_size=args.get('val_batch_size', args.batch_size))
    steps_per_epoch = args.get('steps_per_epoch', len(train_dataloader))
    eval_steps = steps_per_epoch // args.get('evals_per_epoch', 1)
    # print(eval_steps, steps_per_epoch)
    optimizer_class = eval(f'torch.optim.{args.optimizer.tag}')
    # model.evaluate(evaluator, args.save_dir)
    model.train()
    loss_fn.train()
    model.fit(
        train_objectives=[(train_dataloader, loss_fn)],
        evaluator=evaluator,
        epochs=args.epochs,
        scheduler=args.scheduler.tag,
        warmup_steps=args.scheduler.warmup_steps,
        optimizer_class=optimizer_class,
        optimizer_params={'lr': float(args.optimizer.get('lr', 2e-5))},
        weight_decay=args.weight_decay,
        evaluation_steps=eval_steps,
        output_path=args.save_dir,
        save_best_model=True,
        checkpoint_path=os.path.join(args.save_dir, 'checkpoints'),
        checkpoint_save_steps=steps_per_epoch,
        checkpoint_save_total_limit=0
    )     
    return model

def test():
    raise NotImplementedError

def parse_sbert_eval(path):
    import pandas as pd
    return pd.read_csv(path)