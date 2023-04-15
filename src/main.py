import core
import algorithms
import datasets
import metrics
import nets

def main(cfg):
    core.tools.seed_everything(cfg.run.seed)
    cfg.run.debug = cfg.run.get('debug', False)
    algo = f'algorithms.{cfg.run.algorithm}'
    
    if cfg.run.debug : print('=== DEBUG MODE ===')
    for stage in cfg.run.stages:
        print(f'[STAGE={stage}]')
        if stage == 'preprocess':
            topics, items, t2i, i2t, train_folds, test_folds = datasets.proscript.preprocess_data(
                cfg.data.path, splits_path=cfg.data.get('splits_path', None), 
                split_level=cfg.data.split_level, 
                n_folds=cfg.data.n_folds, 
                seed=cfg.run.seed)
            train_data, test_data = train_folds[cfg.data.fold], test_folds[cfg.data.fold]
        elif stage == 'train':
            train_dset = eval(f'datasets.{cfg.data.tag}').get_dataset(
                train_data, topics, items,
                neg_size=cfg.train.neg_size,
                split_level=cfg.data.split_level,
                seed=cfg.run.seed)
            test_dset = eval(f'datasets.{cfg.data.tag}').get_dataset(
                test_data, topics, items,
                neg_size=-1,
                split_level=cfg.data.split_level,
                seed=cfg.run.seed)
            model = nets.nlp.SBERT(cfg.model.arch, device=cfg.train.device)
            eval(algo).train(cfg.train, model, train_dset, test_dset, optimizer=cfg.train.optimizer.tag)
        
        elif stage == 'test':
            eval(algo).test()
        
        elif stage == 'predict':
            eval(algo).predict()
        
        else:
            raise ValueError(f'Run stage={stage} is not valid')

if __name__ == '__main__':
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    main(cfg)