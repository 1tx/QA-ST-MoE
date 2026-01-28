import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import GapfillModel, Transformer_GapfillModel,GapfillModel_NoEmbed,Transformer_NoEmbed,GapfillModel_NoRevin
from models.CNN_LSTM import CNN_LSTM
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from .loss import LearnableWeightedLoss


def train_gap_filler ( args, gap_fill_train_dataset, gap_fill_val_dataset, unique_sites_num, unique_igbp_num, writer ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model
    if args.model_name == 'MOETransformer':
        model = GapfillModel(args, unique_sites_num, unique_igbp_num, "train").to(device)
    elif args.model_name == 'Transformer':
        model = Transformer_GapfillModel(args, unique_sites_num, unique_igbp_num, "train").to(device)
    elif args.model_name == "MOETransformer_NoEmbed":
        model = GapfillModel_NoEmbed(args, "train").to(device)
    elif args.model_name == 'Transformer_NoEmbed':
        model = Transformer_NoEmbed(args,'train').to(device)
    elif args.model_name == 'MOETransformer_NoRevin':
        model = GapfillModel_NoRevin(args,unique_sites_num,unique_igbp_num,'train').to(device)
    elif args.model_name == 'CNN_LSTM':
        model = CNN_LSTM(args).to(device)
    else:
        raise ValueError(f'Not defined model: {args.model_name}')
    torch.cuda.init()
    # 是否是多GPU环境
    gpu_count = torch.cuda.device_count()
    is_main_process = (gpu_count <= 1 or torch.cuda.current_device() == 0)
    # DataLoader
    batch_size = args.batch_size * max(1, gpu_count)
    train_loader = DataLoader(gap_fill_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.num_worker, persistent_workers=False)
    val_loader = DataLoader(gap_fill_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_worker, persistent_workers=False)
    loss = LearnableWeightedLoss(args, args.loss, args.is_QC, args.delta).to(device)
    if gpu_count > 1:
        print(f"Using {gpu_count} GPUs with DataParallel")
        model = DataParallel(model)
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': loss.parameters(), 'lr': args.learning_rate_for_loss}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    # Early stopping
    best_test_loss = float('inf')
    early_stopping_patience = args.patience
    early_stopping_counter = 0
    model_save_dir = os.path.join(args.output_path, args.filling_var, f'exp_{args.exp_num}')
    os.makedirs(model_save_dir, exist_ok=True)
    # 在构建路径之前，先定义好后缀
    loss_suffix = "" if args.enable_QC_loss else "_withoutQCLoss"
    best_model_path = os.path.join(model_save_dir,
                                   f'{args.model_name}_{args.filling_var}_best_model_{args.seq_len}_exp{args.exp_num}'
                                   f'{loss_suffix}.pth')
    checkpoint_path = os.path.join(model_save_dir,
                                   f'{args.model_name}_{args.filling_var}_checkpoint_{args.seq_len}_exp{args.exp_num}'
                                   f'{loss_suffix}.pth')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_loss = checkpoint['best_test_loss']
        early_stopping_counter = checkpoint['early_stopping_counter']
        print(f"Resuming training from epoch {start_epoch} with best test loss: {best_test_loss:.6f}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs} [Train]', leave=False,
                          mininterval=1.0) if is_main_process else train_loader

        for model_input, targets, valid_mask in train_iter:
            for key in model_input:
                model_input[key] = model_input[key].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(model_input)
            error = loss(outputs.float(), targets.float(), valid_mask,args.enable_QC_loss)
            error.backward()
            optimizer.step()
            if is_main_process:
                loss_value = error.item()
                running_loss += loss_value
                train_iter.set_postfix({'Train Loss': f'{loss_value:.6f}'})
        train_loss = running_loss / len(train_loader)
        if is_main_process:
            train_iter.close()
            print(f'\nEpoch [{epoch + 1}/{args.epochs}], Avg Train Loss: {train_loss:.6f}')
            writer.add_scalar('Loss/train', train_loss, epoch)
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_iter = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{args.epochs} [Validation]', leave=False,
                        mininterval=1.0) if is_main_process else val_loader
        with (torch.no_grad()):
            for model_input, targets, valid_mask in val_iter:
                for key in model_input:
                    model_input[key] = model_input[key].to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                valid_mask = valid_mask.to(device, non_blocking=True)
                outputs = model(model_input)
                error = loss(outputs.float(), targets.float(), valid_mask,args.enable_QC_loss)
                if is_main_process:
                    val_loss_sum += error.item()
                    val_iter.set_postfix({'Validation Loss': f'{error.item():.6f}'})
        test_loss = val_loss_sum / len(val_loader)
        if is_main_process:
            val_iter.close()
            print(f'Epoch [{epoch + 1}/{args.epochs}], Avg Validation Loss: {test_loss:.6f}')
            writer.add_scalar('Loss/Validation', test_loss, epoch)
        scheduler.step(test_loss)
        # 保存断点
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_test_loss': best_test_loss,
            'early_stopping_counter': early_stopping_counter,
            'args': args,
        }
        torch.save(checkpoint_state, checkpoint_path)
        if is_main_process:
            print(f'Checkpoint saved for epoch {epoch + 1} to {checkpoint_path}')

        # 早停逻辑
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            if is_main_process:
                print(f'Best model saved (Loss: {best_test_loss:.6f})')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if is_main_process:
                print(f' No improvement ({early_stopping_counter}/{early_stopping_patience})')

        if early_stopping_counter >= early_stopping_patience:
            if is_main_process:
                print(f' Early stopping at epoch {epoch + 1}')
            break
    if is_main_process:
        print(f'\nTraining completed. Best model saved to: {best_model_path}')
        writer.close()
    return
