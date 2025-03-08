import os
import torch
import wandb
import torchmetrics
from pytorch_lightning import seed_everything
import json
from utils import parse_arguments
from utils import get_categories_list, gettime_str, print_trainable_parameters, get_weight_decay_params, \
    create_directory_if_not_exists
from dataset.dataset import Dataset
from models.model_factory import get_model_and_transform
from wrapper.projection_head import ProjectionHead
from wrapper.mlp_head import MLPHead
from wrapper.lora_wrapper import CLIPWithLoRA
from train import train_epoch, valid_epoch
from utils import create_train_transform

# export http_proxy=http://localhost:7890
# export https_proxy=http://localhost:7890

def main():
    args = parse_arguments()
    seed_everything(args.seed)

    # Setup logging and naming
    name_message = f"{args.mode}_{args.model_type}_{args.variant}_bs{args.batchsize}_epochs{args.epochs}_lr{args.lr}_{gettime_str()}"
    log_name = f"{name_message}_logs.txt"
    pth_name = f"{name_message}.pth"
    cm_name = f"{name_message}_cm.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load categories and create diagnosis_map
    csv_path_all = './dataset/nih6x200.csv'
    categories = get_categories_list(csv_path_all)
    diagnosis_map = {str(categories[i]): i for i in range(len(categories))}

    # Load model and transforms
    base_model, val_transform, embedding_dim = get_model_and_transform(args.model_type, args.variant, device)
    train_transform = create_train_transform(val_transform)

    # Setup datasets and loaders
    cxr_filepath = '/mnt/disk/sxyy/ISBI_2025_proj/dataset/images'
    train_dataset = Dataset(train_transform, './dataset/nih6x200_train.csv', 'train', cxr_filepath)
    val_dataset = Dataset(val_transform, './dataset/nih6x200_val.csv', 'val', cxr_filepath)
    test_dataset = Dataset(val_transform, './dataset/nih6x200_test.csv', 'test', cxr_filepath)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4,
                                              pin_memory=True)

    # Configure model based on mode
    if args.mode == 'ft':
        for param in base_model.parameters():
            param.requires_grad = True
    else:
        for param in base_model.parameters():
            param.requires_grad = False

    if args.mode in ['lp', 'ft']:
        model = ProjectionHead(base_model, embedding_dim, 1024, len(diagnosis_map))
    elif args.mode == 'mlp':
        model = MLPHead(base_model, embedding_dim, 1024, len(diagnosis_map))
    elif args.mode == 'lora':
        is_vit = (args.model_type == 'clip' and 'ViT' in args.variant) or \
                 (args.model_type == 'medclip' and args.variant == 'vit') or \
                 (args.model_type == 'biomedclip')
        model = CLIPWithLoRA(base_model, is_vit, embedding_dim, 1024, len(diagnosis_map))
    # elif args.mode == 'zero-shot':
    #     model = base_model  # Placeholder; zero-shot logic would need additional implementation
    model.to(device)

    print_trainable_parameters(model)
    get_weight_decay_params(model)

    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0, last_epoch=-1)

    metrics = {
        'AUROC': torchmetrics.AUROC(task="multiclass", num_classes=len(diagnosis_map), average="macro"),
        'Accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(
            device),
        'Precision': torchmetrics.Precision(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(
            device),
        'Recall': torchmetrics.Recall(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(device),
        'F1Score': torchmetrics.F1Score(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(device)
    }

    # Initialize WandB
    wandb.init(project=name_message, id=name_message, config=args, resume='allow')

    # Training loop
    best_loss = float('inf')
    log_path = args.path
    create_directory_if_not_exists(log_path)

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, train_report = train_epoch(model, train_loader, optimizer, criterion, lr_scheduler, metrics, device)
        model.eval()
        with torch.no_grad():
            valid_loss, valid_report = valid_epoch(model, valid_loader, criterion, metrics.copy(), device, mode='val',
                                                   log_path=log_path, cm_file=cm_name)
            wandb.log({'train_loss': train_loss.avg, 'val_loss': valid_loss.avg, **train_report, **valid_report})

            with open(os.path.join(log_path, log_name), 'a') as f:
                f.write(json.dumps(train_report) + '\n')
                f.write(json.dumps(valid_report) + '\n')

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.state_dict(), os.path.join(log_path, pth_name))
                print(f"Best loss updated: {best_loss}")

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(log_path, pth_name)))
    model.eval()
    with torch.no_grad():
        test_loss, test_report = valid_epoch(model, test_loader, criterion, metrics.copy(), device, mode='test',
                                             log_path=log_path, cm_file=cm_name)
        wandb.log({'test_loss': test_loss.avg, **test_report})
        with open(os.path.join(log_path, log_name), 'a') as f:
            f.write(json.dumps(test_report) + '\n')
        print(f"Test report: {test_report}")


if __name__ == "__main__":
    main()