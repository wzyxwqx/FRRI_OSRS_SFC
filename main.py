import argparse
import os
import time
import datetime
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import script_utils, evaluation
from datasets import get_datasets
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


def main():
    args = create_argparser().parse_args()
    assert args.task in ['train','test','synthesis']
    print('[Init] with args:',args)
    device = args.device
    batch_size = args.batch_size
    data_shape = args.data_shape
    devices_range = args.devices_range
    model_classes_num = devices_range['known_max'] - devices_range['known_min'] + 1 
    model_classes_num += 1 # N+1
    if args.task == 'synthesis':
        da={'pa':2.3}
        print(f'[Init] synthesize data with pa paramater:{2.3}')
    else:
        da={}
    print(f'[Init] Device:{device}, model_classes_num:{model_classes_num}, devices_range:', devices_range)
    try:
        model = script_utils.get_model_from_args(args, num_classes=model_classes_num, data_shape=data_shape).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []
        early_stop_count = 0
        epoch_start = 1
        if args.ckpt_path is not None:
            model, optimizer, epoch_start, datasets_indices = load_ckpt(args.ckpt_path, model, optimizer, args)
            train_dataset, val_dataset, test_dataset, _, datasets_indices = get_datasets(dataset_path = args.dataset_path, devices_range=devices_range, data_shape=data_shape, datasets_indices=datasets_indices, samples_file=args.samples_file, da=da)
        else:
            train_dataset, val_dataset, test_dataset, _, datasets_indices = get_datasets(dataset_path = args.dataset_path,devices_range=devices_range, data_shape=data_shape, datasets_indices=None, samples_file=args.samples_file, da=da)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=8)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=8)
        
        if args.task == 'train':
            last_epoch_time = time.time()
            for epoch in range(epoch_start, args.max_epoches + 1):
                train_loss, train_acc, _, _, _, _= epoch_loop(train_loader, optimizer, model, args,'train')
                train_acc_list.append(train_acc)
                train_loss_list.append(train_loss)
                train_end_time = time.time()
                train_time = train_end_time - last_epoch_time

                val_loss, val_acc, scores, pred_softmax, score_softmax, labels = epoch_loop(val_loader, optimizer, model, args, 'val')
                val_acc_list.append(val_acc)
                val_end_time = time.time()
                val_time = val_end_time - train_end_time
                last_epoch_time = val_end_time

                print(f'[train] Epoch:{epoch:<4d} Train Acc:{train_acc:<8.2%} Val Acc:{val_acc:<8.2%} || Train Loss:{train_loss:<21} Val Loss:{val_loss:<21} || Train time:{train_time / 60:<6.2f} min. Val time:{val_time / 60:<6.2f} min')
                if len(val_loss_list) == 0 or val_loss < min(val_loss_list):
                    early_stop_count = 0
                    best_epoch = epoch
                    best_val_acc = val_acc
                    best_train_acc = train_acc
                    checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'datasets_indices': datasets_indices,
                                  'devices_range': devices_range}
                    ckpt_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-model.pth"
                    torch.save(checkpoint, ckpt_filename)
                else:
                    early_stop_count += 1
                    if early_stop_count >= 60:
                        print('Training early stop!')
                        break
                val_loss_list.append(val_loss)
            model.load_state_dict(checkpoint['state_dict'])
            print(f'Final epoch:{epoch}, best val loss:{min(val_loss_list):8}, best val accuracy:{best_val_acc:>8.2%}, best train accuracy:{best_train_acc:>8.2%} in epoch:{best_epoch}, ')
            test_loop(test_loader, model, args, devices_range)
        elif args.task == 'test':
            assert args.ckpt_path is not None
            test_loop(test_loader, model, args, devices_range)
        elif args.task == 'synthesis':
            synthesis_loop(train_loader, args, devices_range, da)

    except KeyboardInterrupt:
        print("Keyboard interrupt, run finished early")

def load_ckpt(ckpt_path, model, optimizer, args):
    if os.path.isfile(ckpt_path):
        print("Continue training!\n=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)

        # only load parameters exist in current model
        current_model_dict = model.state_dict()
        keys_are_equal = set(current_model_dict.keys()) == set(checkpoint['state_dict'].keys())
        if not keys_are_equal:
            print('Old CKPT is not totally same with current CKPT, load part of the old one.')
            # do not load optimizer
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        old_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in current_model_dict}
        current_model_dict.update(old_state_dict)
        model.load_state_dict(current_model_dict)
        epoch_start = checkpoint['epoch'] + 1
        datasets_indices = checkpoint['datasets_indices']
        if 'devices_range' in checkpoint.keys():
            devices_range = checkpoint['devices_range']
            print('Devices range in checkpoint:', devices_range)
        print("Loaded checkpoint, epoch start at:", epoch_start)
        print("Load state keys len:", len(old_state_dict.keys()))
        return model, optimizer, epoch_start, datasets_indices
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))

def epoch_loop(dataloader, optimizer, model, args, phase):
    total_loss, correct, data_index = 0, 0, 0
    labels, scores = [], []
    pred_softmax, score_softmax = [], []
    if phase == 'train':
        model.train()
    else:
        model.eval()
    size = 0
    torch.set_grad_enabled(phase == 'train')
    for batchidx, (X, y) in enumerate(dataloader):
        X = X.to(args.device)
        y = y.to(args.device)
        size += X.size(0)
        out = model(X)
        if phase == 'val':
            labels.append(y[:, 0])
            scores.append(out[0])
        loss = model.get_loss(out, y)
        # Backpropagation
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out[0].argmax(1) == (y[:, 0])).type(torch.float).sum().item()
    total_loss /= size
    acc = correct / size

    if phase == 'val':
        scores = torch.cat(scores, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        for score in scores:
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            ss = softmax(np.array(score.ravel()))
            pred_softmax.append(ss.argmax())
            score_softmax.append(ss)
    return total_loss, acc, scores, pred_softmax, score_softmax, labels

def synthesis_loop(train_dataloader, args, devices_range, da):
    class_num_kkc = devices_range['known_max'] - devices_range['known_min'] + 1
    batch_size = args.batch_size
    num_samples = len(train_dataloader.dataset)
    batch_num = num_samples // batch_size + 1
    X_shape = (num_samples, args.data_shape[0], args.data_shape[1], args.data_shape[2])
    # prepare samples saved files
    samples_save_folder = f'samples/'
    if not os.path.exists(samples_save_folder):
        os.mkdir(samples_save_folder)
    da_type = list(da.keys())[0]
    da_v = da[da_type]
    samples_save_file = samples_save_folder + f'/multi-label-{da_type}-{da_v}-{args.run_name}-{num_samples}.h5'
    print(f'Samples save to ', script_dir + '/' + samples_save_file)
    fdes = h5py.File(script_dir + '/' + samples_save_file, "w")
    Xdes = fdes.create_dataset(f'X', X_shape, dtype=np.float32)
    Ydes = fdes.create_dataset(f'Y', (num_samples,), dtype=np.int32)
    start_idx = 0
    with torch.no_grad():
        for batchidx, (X, y) in enumerate(train_dataloader):
            X = X.numpy()
            y = y.numpy()
            num_sample_this_batch = batch_size if batchidx < batch_num - 1 else (num_samples % batch_size)
            Xdes[start_idx:start_idx + num_sample_this_batch] = X[:num_sample_this_batch]
            Ydes[start_idx:start_idx + num_sample_this_batch] = y + class_num_kkc
            start_idx += num_sample_this_batch
    fdes.close()
    assert start_idx == num_samples
    print(f'finish sample, save {num_samples} data, lables of syn data: [{class_num_kkc}, {2*class_num_kkc-1}]')

def test_loop(test_dataloader, model, args, devices_range):
    '''
        For method in SoftMax OpenMax
    '''
    class_num_kkc = devices_range['known_max'] - devices_range['known_min'] + 1
    model.eval()
    size = 0
    scores_N, labels = [], []
    with torch.no_grad():
        for batchidx, (X, y) in enumerate(test_dataloader):
            X = X.to(args.device)
            y = y.to(args.device)
            out = model(X)
            out = out[0]
            out_cal = out[:, :class_num_kkc]
            scores_N.append(out_cal)
            size += X.size(0)
            labels.append(y)

    scores_N = torch.cat(scores_N, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    closeset_indices = labels < class_num_kkc
    labels[labels >= class_num_kkc] = class_num_kkc  # N+1 class
    labels_closeset = labels[closeset_indices]
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    pred_softmax_N, pred_softmax_threshold_N = [], []
    score_softmax_N, score_softmax_threshold_N = [], []
    for score in scores_N:
        ss = softmax(np.array(score.ravel()))
        pred_softmax_N.append(ss.argmax())
        pred_softmax_threshold_N.append(ss.argmax() if np.max(ss) >= args.classification_threshold else class_num_kkc)
        score_softmax_N.append(ss)
        sst = np.concatenate((ss, [ss.sum() - ss.max()])) / (ss.sum() * 2 - ss.max())
        score_softmax_threshold_N.append(sst)
    score_softmax_closeset_N = [softmax(s) for s, flag in zip(scores_N, closeset_indices) if flag]
    pred_softmax_closeset_N = np.array(pred_softmax_N)[closeset_indices].tolist()
    print("Evaluation...")

    eval_softmax = evaluation.Evaluation(pred_softmax_closeset_N, labels_closeset, score_softmax_closeset_N)
    print(f"Close set accuracy is %.3f" % (eval_softmax.accuracy))
    print(f"Close set area_under_roc_weighted is %.3f" % (eval_softmax.area_under_roc_weighted))
    print(f"Close set area_under_roc_per_class is ",
          " ".join(f"{num:.3f}" for num in eval_softmax.area_under_roc_per_class))
    print(f"_________________________________________")
    
    eval_softmax_threshold = evaluation.Evaluation(pred_softmax_threshold_N, labels, score_softmax_threshold_N)
    print(f"SFCR {args.classification_threshold} accuracy is %.3f" % (
        eval_softmax_threshold.accuracy))
    print(f"SFCR {args.classification_threshold} area_under_roc_weighted is %.3f" % (
        eval_softmax_threshold.area_under_roc_weighted))
    print(f"SFCR {args.classification_threshold} area_under_roc_per_class is ",
            " ".join(f"{num:.3f}" for num in eval_softmax_threshold.area_under_roc_per_class))
    print(f"_________________________________________")

def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    defaults = dict(
        ckpt_path= None,  # None
        task='train',  # 'train' or 'test' or 'synthesis'
        learning_rate=1e-4,
        batch_size=1024,
        device=device,
        max_epoches=500,
        log_rate=1,
        checkpoint_rate=5,
        log_dir="models",
        run_name=run_name,

        dataset_path = None,
        data_shape=(128, 64, 2),
        devices_range={'known_min': 0, 'known_max': 29,
                       'val_uuc_min': None, 'val_uuc_max': None,
                       'test_uuc_min': 30, 'test_uuc_max': 39},
        samples_file = None,
        classification_threshold = 0.9,
        gamma = 0.001,
    )
    
    if defaults['task'] == 'synthesis':
        defaults['data_shape'] = (1, defaults['data_shape'][0] * defaults['data_shape'][1], 2)
    defaults['project_name'] = 'SFCR'
    if not os.path.exists(defaults['log_dir']):
        os.mkdir(defaults['log_dir'])
    print(f'Method: SFCR, task:', defaults['task'])
    print(f'Ckpt save path:', defaults['log_dir'], ', runname:', run_name)
    if not os.path.exists(defaults['log_dir']):
        os.mkdir(defaults['log_dir'])
    parser = argparse.ArgumentParser()

    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()