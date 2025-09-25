import os
import json
import torch
import random
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from model import SentimentClassifier
from sklearn.metrics import f1_score
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers import BartTokenizer
from utils import parse_args, Log, load_data, create_data_loader, train_epoch, eval_model ,format_eval_output

if __name__ == '__main__':
    args = parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Logger
    logger = Log(
        os.path.join('./log','0805',str(args.triple_number)),
        f"{args.dataset}_{args.image_feature}_{args.LEARNING_RATE}"
    ).get_logger()
    logger.info(args)

    # Tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    tokenizer.add_special_tokens({
        'additional_special_tokens': [
            "<image>", "<caption>", "</caption>", "<tweet>", "</tweet>",
            "<target>", "</target>", "<triple>", "</triple>", "<ts>"
        ]
    })

    # Seeds
    if args.RANDOM_SEEDS != -1:
        random.seed(args.RANDOM_SEEDS)
        np.random.seed(args.RANDOM_SEEDS)
        torch.manual_seed(args.RANDOM_SEEDS)
        torch.cuda.manual_seed(args.RANDOM_SEEDS)
        torch.cuda.manual_seed_all(args.RANDOM_SEEDS)
        torch.backends.cudnn.deterministic = True

    # Load data
    train_df, val_df, test_df, image_captions, imageid2triple = load_data(args)

    # Data loaders
    train_data_loader = create_data_loader(
        train_df, tokenizer, args.MAX_LEN, args.BATCH_SIZE,
        image_captions, imageid2triple, args.dataset, args.triple_number
    )
    val_data_loader = create_data_loader(
        val_df, tokenizer, args.MAX_LEN, args.BATCH_SIZE,
        image_captions, imageid2triple, args.dataset, args.triple_number
    )
    test_data_loader = create_data_loader(
        test_df, tokenizer, args.MAX_LEN, args.BATCH_SIZE,
        image_captions, imageid2triple, args.dataset, args.triple_number
    )

    # Model
    model = SentimentClassifier(args, tokenizer).to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.LEARNING_RATE)
    total_steps = len(train_data_loader) * args.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.NUM_WARMUP_STEPS, num_training_steps=total_steps
    )

    # Loss
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Training
    best_acc = 0.
    best_f1_with = 0.
    best_f1 = 0.
    best_acc_with = 0.

    for epoch in range(args.EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{args.EPOCHS}")

        # Train
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, scheduler, len(train_df), tokenizer=tokenizer, device=device)
        logger.info(f"Train loss {train_loss} accuracy {train_acc}")

        # Validation
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, len(val_df), tokenizer=tokenizer, device=device)
        logger.info(f"Val   loss {val_loss} accuracy {val_acc}")

        # Test
        test_acc, _, detailed_results = eval_model(model, test_data_loader, loss_fn, len(test_df), detailed_results=True, tokenizer=tokenizer, device=device)
        macro_f1 = f1_score(detailed_results.label, detailed_results.prediction, average="macro")
        logger.info(f"TEST ACC = {test_acc}\nMACRO F1 = {macro_f1}")

        # Track best results
        if test_acc > best_acc:
            best_acc = test_acc
            best_f1_with = macro_f1
            best_epoch_acc = epoch

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_acc_with = test_acc
            best_epoch_f1 = epoch

        logger.info("----- best result now: -----")
        logger.info(f"acc!!!  [epoch:{best_epoch_acc}] TEST ACC = {best_acc}   MACRO F1 = {best_f1_with}")
        logger.info(f"f1!!!   [epoch:{best_epoch_f1}] TEST ACC = {best_acc_with}  MACRO F1 = {best_f1}")

        # Save model
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
