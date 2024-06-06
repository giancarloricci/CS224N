'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random
import numpy as np
import argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from gradient_surgery import PCGrad

from tokenizer import BertTokenizer

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        self.ln_sentiment = nn.Linear(
            in_features=config.hidden_size, out_features=5)
        self.ln_paraphrase = nn.Linear(
            in_features=config.hidden_size, out_features=1)
        self.ln_similarity = nn.Linear(
            in_features=config.hidden_size, out_features=1)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask)
        out = out["pooler_output"]
        out = self.dropout(out)
        return out

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        out = self.forward(input_ids, attention_mask)
        out = self.ln_sentiment(out)
        return out

    def combined_inputs(self, input_ids_1, attention_mask_1,
                        input_ids_2, attention_mask_2):
        sep_token_id = torch.tensor(
            [self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)
        input_id = torch.cat(
            (input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(
            batch_sep_token_id), attention_mask_2, torch.ones_like(batch_sep_token_id)), dim=1)
        return input_id, attention_mask

    def combined_inputs(self, input_ids_1, attention_mask_1,
                        input_ids_2, attention_mask_2):
        sep_token_id = torch.tensor(
            [self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)
        input_id = torch.cat(
            (input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(
            batch_sep_token_id), attention_mask_2, torch.ones_like(batch_sep_token_id)), dim=1)
        return input_id, attention_mask

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        input_id, attention_mask = self.combined_inputs(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        out = self.forward(input_id, attention_mask)
        return self.ln_paraphrase(out)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        input_id, attention_mask = self.combined_inputs(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        out = self.forward(input_id, attention_mask)
        return self.ln_similarity(out)

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


class Task:
    def __init__(self, dataloader, predictor, loss_function, single_sentence, linear_layer):
        self.dataloader = dataloader
        self.predictor = predictor
        self.loss_function = loss_function
        self.single_sentence = single_sentence
        self.linear_layer = linear_layer


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(
        sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    def get_input_labels(batch, single_sentence=True):
        if single_sentence:
            b_ids, b_mask, b_labels = batch['token_ids'], batch['attention_mask'], batch['labels']
            b_ids, b_mask, b_labels = map(
                lambda x: x.to(device), (b_ids, b_mask, b_labels))
            return (b_ids, b_mask), b_labels
        else:
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                batch['token_ids_1'], batch['attention_mask_1'],
                batch['token_ids_2'], batch['attention_mask_2'],
                batch['labels']
            )
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = map(
                lambda x: x.to(device), (b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels))
            return (b_ids_1, b_mask_1, b_ids_2, b_mask_2), b_labels

    if args.load_model:
        saved = torch.load(args.load_filepath)
        config = saved['model_config']
        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.load_filepath}")

    else:
        # Init model.
        config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                  'num_labels': num_labels,
                  'hidden_size': 768,
                  'data_dir': '.',
                  'fine_tune_mode': args.fine_tune_mode,
                  "n_hidden_layers": args.n_hidden_layers
                  }

        config = SimpleNamespace(**config)

        model = MultitaskBERT(config)
        model = model.to(device)
        # if args.use_smart == 0:
        #     bpp = BPP(model, beta=0.8, mu=1)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    if args.gradient_surgery:
        optimizer = PCGrad(optimizer)

    task_sst = Task(
        sst_train_dataloader,
        model.predict_sentiment,
        lambda logits, b_labels: F.cross_entropy(
            logits, b_labels.view(-1), reduction='sum') / args.batch_size,
        True,
        model.ln_sentiment
    )

    def conditional_loss(logits, b_labels):
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), b_labels.float(), reduction='sum') / args.batch_size
        if args.weight:
            loss *= args.weight_factor
        return loss

    task_para = Task(
        para_train_dataloader,
        model.predict_paraphrase,
        conditional_loss,
        False,
        model.ln_paraphrase
    )

    task_sts = Task(
        sts_train_dataloader,
        model.predict_similarity,
        lambda logits, b_labels: F.mse_loss(
            logits.view(-1), b_labels.view(-1).float(), reduction='sum') / args.batch_size,
        False,
        model.ln_similarity
    )

    tasks = [task_sst, task_para, task_sts]
    num_tasks = len(tasks)
    best_dev_acc = 0

    def apply_smart(task, predict_args, logits, b_labels, model):
        if args.use_smart == 0:
            raise NotImplementedError
        else:
            return task.loss_function(logits, b_labels)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.gradient_surgery:
            if args.surgery_mode == 0:
                if args.sampling == 0:
                    min_batches = min(len(task.dataloader) for task in tasks)
                    dataloader_iters = [iter(task.dataloader) for task in tasks]

                    with tqdm(total=min_batches) as pbar:
                        for _ in range(min_batches):
                            losses = []

                            for i, task in enumerate(tasks):
                                optimizer.zero_grad()
                                try:
                                    batch = next(dataloader_iters[i])
                                except StopIteration:
                                    continue

                                predict_args, b_labels = get_input_labels(
                                    batch, task.single_sentence)
                                logits = task.predictor(*predict_args)
                                loss = apply_smart(task, predict_args,
                                                logits, b_labels, model)
                                losses.append(loss)

                            assert len(losses) == num_tasks
                            optimizer.pc_backward(losses)
                            optimizer.step()

                            train_loss += sum(loss.item() for loss in losses)
                            num_batches += 1
                            pbar.update(1)
                            
                elif args.sampling == 1:
                    num_epochs = args.epochs
                    num_tasks = len(tasks)
                    dataset_sizes = np.array([len(task.dataloader.dataset) for task in tasks])
                      
                    batches_per_epoch = args.num_batches
                    alpha = 1 - 0.8 * (epoch / (num_epochs - 1))
                        
                    task_probabilities = dataset_sizes ** alpha
                    task_probabilities /= task_probabilities.sum()
                    dataloader_iters = [iter(task.dataloader) for task in tasks]
                    if args.downsample == 1:
                        dataloader = tasks[0].dataloader
                        size = args.downsample_size
                      
                        dataloader = torch.utils.data.DataLoader(
                                dataloader.dataset[:size],
                                batch_size=dataloader.batch_size,
                                shuffle=True,
                                num_workers=dataloader.num_workers,
                                collate_fn=dataloader.collate_fn,
                                drop_last=dataloader.drop_last
                        )
                        dataloader_iters[0] = iter(dataloader)

                    with tqdm(total=batches_per_epoch) as pbar:
                        for _ in range(batches_per_epoch):
                            losses = []
                                
                            for _ in range(num_tasks):
                                optimizer.zero_grad()
                                    
                                task_index = np.random.choice(num_tasks, p=task_probabilities)
                                selected_task = tasks[task_index]
                                
                                try:
                                    batch = next(dataloader_iters[task_index])
                                except StopIteration:
                                    # If the current task is exhausted, try other tasks
                                    for other_task_index in range(num_tasks):
                                        if other_task_index != task_index:  # Skip the current task
                                            try:
                                                batch = next(dataloader_iters[other_task_index])
                                                selected_task = tasks[other_task_index]
                                                break  # Exit the loop if a batch is successfully obtained
                                            except StopIteration:
                                                continue
                                    else:
                                        raise ValueError("All tasks have exhausted their dataloaders")

                                    
                                predict_args, b_labels = get_input_labels(batch, selected_task.single_sentence)
                                logits = selected_task.predictor(*predict_args)
                                loss = apply_smart(selected_task, predict_args, logits, b_labels, model)
                                losses.append(loss)
                                
                            assert len(losses) == num_tasks
                            optimizer.pc_backward(losses)
                            optimizer.step()
                                
                            train_loss += sum(loss.item() for loss in losses)
                            num_batches += 1
                            pbar.update(1)
                            
                elif args.sampling == 2:
                    dataset_sizes = [len(task.dataloader.dataset) for task in tasks]
                    total_samples = sum(dataset_sizes)
                    probabilities = [size / total_samples for size in dataset_sizes]
                    dataloader_iters = [iter(task.dataloader) for task in tasks]
                    batches_per_epoch = args.num_batches
                    with tqdm(total=batches_per_epoch) as pbar:
                        for _ in range(batches_per_epoch):
                            losses = []
                                
                            for _ in range(num_tasks):
                                optimizer.zero_grad()
                                task_index = np.random.choice(num_tasks, p=probabilities)
                                selected_task = tasks[task_index]
                            
                                try:
                                    batch = next(dataloader_iters[task_index])
                                except StopIteration:
                                    continue
                                
                                predict_args, b_labels = get_input_labels(batch, selected_task.single_sentence)
                                logits = selected_task.predictor(*predict_args)
                                loss = apply_smart(selected_task, predict_args, logits, b_labels, model)
                                losses.append(loss)

                            assert len(losses) == num_tasks
                            optimizer.pc_backward(losses)
                            optimizer.step()
                            
                            train_loss += sum(loss.item() for loss in losses)
                            num_batches += 1
                            pbar.update(1)
                                    
            elif args.surgery_mode == 1:
                dataloader_iters = [iter(task.dataloader) for task in tasks]

                if args.continue_surg:
                    min_batches = min(len(task.dataloader) for task in tasks)
                    with tqdm(total=min_batches) as pbar:
                        for _ in range(min_batches):
                            losses = []

                            for i, task in enumerate(tasks):
                                optimizer.zero_grad()
                                try:
                                    batch = next(dataloader_iters[i])
                                except StopIteration:
                                    continue

                                predict_args, b_labels = get_input_labels(
                                    batch, task.single_sentence)
                                logits = task.predictor(*predict_args)
                                loss = apply_smart(task, predict_args,
                                                   logits, b_labels, model)
                                losses.append(loss)

                            assert len(losses) == num_tasks
                            optimizer.pc_backward(losses)
                            optimizer.step()

                            train_loss += sum(loss.item() for loss in losses)
                            num_batches += 1
                            pbar.update(1)

                if args.surg_debug:
                    # disable everything but us
                    if args.disable_bert == 0:
                        for param in model.bert.parameters():
                            param.requires_grad = False
                        for param in model.ln_sentiment.parameters():
                            param.requires_grad = False
                        for param in model.ln_similarity.parameters():
                            param.requires_grad = False

                    if args.disable_bert == 1:
                        for param in model.ln_sentiment.parameters():
                            param.requires_grad = False

                    if args.disable_bert == 2:
                        for param in model.bert.parameters():
                            param.requires_grad = False
                        for param in model.ln_sentiment.parameters():
                            param.requires_grad = False

                    dataloader_iter_para = iter(tasks[1].dataloader)
                    para_batches = args.para_size
                    if args.use_max_para:
                        para_batches = len(dataloader_iter_para) - num_batches

                    for _ in tqdm(range(para_batches), desc=f'train-{epoch}', disable=TQDM_DISABLE):
                        optimizer.zero_grad()
                        try:
                            batch = next(dataloader_iter_para)
                        except StopIteration:
                            break

                        predict_args, b_labels = get_input_labels(
                            batch, tasks[1].single_sentence)
                        logits = tasks[1].predictor(*predict_args)
                        loss = apply_smart(tasks[1], predict_args,
                                           logits, b_labels, model)
                        loss = loss * args.penalty_factor2
                        if args.pc_test:
                            loss.backward()
                        else:
                            losses = [loss]
                            optimizer.pc_backward(losses)

                        optimizer.step()
                        train_loss += loss.item()
                        num_batches += 1

                    for param in model.parameters():
                        param.requires_grad = True
                    for param in model.ln_sentiment.parameters():
                        param.requires_grad = True
                    for param in model.ln_similarity.parameters():
                        param.requires_grad = True

                    if args.do_sentiment:
                        for param in model.bert.parameters():
                            param.requires_grad = False
                        for param in model.ln_paraphrase.parameters():
                            param.requires_grad = False
                        for param in model.ln_similarity.parameters():
                            param.requires_grad = False

                        dataloader_iter_sim = dataloader_iters[0]
                        sim_batches = len(dataloader_iter_sim) - min_batches

                        for _ in tqdm(range(sim_batches), desc=f'train-{epoch}', disable=TQDM_DISABLE):
                            optimizer.zero_grad()
                            try:
                                batch = next(dataloader_iter_sim)
                            except StopIteration:
                                break

                            predict_args, b_labels = get_input_labels(
                                batch, tasks[0].single_sentence)
                            logits = tasks[0].predictor(*predict_args)
                            loss = apply_smart(tasks[0], predict_args,
                                               logits, b_labels, model)

                            loss = loss * args.penalty_factor2
                            if args.pc_test:
                                loss.backward()
                            else:
                                losses = [loss]
                                optimizer.pc_backward(losses)

                            optimizer.step()
                            train_loss += loss.item()
                            num_batches += 1

                        for param in model.parameters():
                            param.requires_grad = True
                        for param in model.ln_sentiment.parameters():
                            param.requires_grad = True
                        for param in model.ln_similarity.parameters():
                            param.requires_grad = True
        else:
            for task in tasks:
                for batch in tqdm(task.dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                    optimizer.zero_grad()
                    predict_args, b_labels = get_input_labels(
                        batch, task.single_sentence)
                    logits = task.predictor(*predict_args)
                    loss = apply_smart(task, predict_args,
                                       logits, b_labels, model)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    num_batches += 1

        train_loss = train_loss / (num_batches)
        print("training done")

        if args.intermediate_eval:
            if args.include_train:
                train_acc, train_f1, *_ = model_eval_multitask(
                    sst_train_dataloader, para_train_dataloader, sts_train_dataloader,
                    model, device
                )

            dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
                dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
                dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                                      para_dev_dataloader,
                                                                                      sts_dev_dataloader, model, device)

            dev_acc = (dev_sentiment_accuracy +
                       dev_paraphrase_accuracy + (0.5 * dev_sts_corr + 0.5)) / 3
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)
            if args.include_train:
                print(
                    f"Epoch {epoch} : train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
            else:
                print(
                    f"Epoch {epoch} : dev acc :: {dev_acc :.3f}")

        if args.always_save:
            save_model(model, optimizer, args, config, args.filepath)

    if not args.intermediate_eval:
        save_model(model, optimizer, args, config, args.filepath)


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test, args.para_test,
                                args.sts_test, split='test')

        sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev, args.para_dev,
                                args.sts_dev, split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(
            sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                                  para_dev_dataloader,
                                                                                  sts_dev_dataloader, model, device)

        if args.include_test:
            test_sst_y_pred, \
                test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        if args.include_test:
            with open(args.sst_test_out, "w+") as f:
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                    f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        if args.include_test:
            with open(args.para_test_out, "w+") as f:
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(test_para_sent_ids, test_para_y_pred):
                    f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        if args.include_test:
            with open(args.sts_test_out, "w+") as f:
                f.write(f"id \t Predicted_Similiary \n")
                for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                    f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str,
                        default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str,
                        default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str,
                        default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str,
                        default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str,
                        default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str,
                        default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str,
                        default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str,
                        default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str,
                        default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str,
                        default="predictions/sts-dev-output.csv")
    
    parser.add_argument("--sts_test_out", type=str,
                        default="predictions/sts-test-output.csv")

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--no_train", type=str,
                        help="learning rate", default="False")

    parser.add_argument("--intermediate_eval", type=str,
                        help="evaluate every epoch", default="True")

    parser.add_argument("--use_smart", type=int,
                        help="use SMART optimization", default=1)
    
    parser.add_argument("--weight", type=str,
                        help="weight", default="true")
    
    parser.add_argument("--sampling", type=int,
                        help="weight", default=0)
    
    parser.add_argument("--num_batches", type=int,
                        help="weight", default=12000)
    
    parser.add_argument("--weight_factor", type=float,
                        help="weight", default=1)
      
    parser.add_argument("--downsample", type=int,
                        help="weight", default=1)
    
    parser.add_argument("--downsample_size", type=int,
                        help="weight", default=10000)

    parser.add_argument("--train", type=str,
                        help="use SMART optimization", default="true")

    parser.add_argument("--smart_bpp", type=int,
                        help="use SMART optimization", default=0)

    parser.add_argument("--gradient_surgery", type=str,
                        help="use gradient surgery", default="True")

    parser.add_argument("--always_save", type=str,
                        help="save model after each epoch", default="False")

    parser.add_argument("--surgery_mode", type=int,
                        help="surgery mode", default=1)

    parser.add_argument("--para_size", type=int,
                        help="how much of paraphrase to use", default=1000)

    parser.add_argument("--disable_bert", type=int,
                        help="toggle disable when not all, just one, or not at all", default=0)

    parser.add_argument("--penalty_factor1", type=float,
                        help="", default=0.3)

    parser.add_argument("--penalty_factor2", type=float,
                        help="", default=1)

    parser.add_argument("--surg_debug", type=str,
                        help="", default="True")

    parser.add_argument("--include_test", type=str,
                        help="", default="True")

    parser.add_argument("--pc_test", type=str,
                        help="", default="True")

    parser.add_argument("--use_max_para", type=str,
                        help="", default="True")

    parser.add_argument("--do_sentiment", type=str,
                        help="", default="False")

    parser.add_argument("--include_train", type=str,
                        help="", default="False")

    parser.add_argument("--cutoff", type=int,
                        help="", default=0)

    parser.add_argument("--load_model", type=str,
                        help="", default="False")

    parser.add_argument("--load_filepath", type=str,
                        help="", default="full-model-9-1e-05-multitask.pt")

    parser.add_argument("--continue_surg", type=str,
                        help="", default="false")

    parser.add_argument("--n_hidden_layers", type=int,
                        help="", default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.intermediate_eval = args.intermediate_eval.lower() == "true"
    args.weight = args.weight.lower() == "true"
    args.include_train = args.include_train.lower() == "true"
    args.load_model = args.load_model.lower() == "true"
    args.continue_surg = args.continue_surg.lower() == "true"
    args.surg_debug = args.surg_debug.lower() == "true"
    args.include_test = args.include_test.lower() == "true"
    args.train = args.train.lower() == "true"
    args.gradient_surgergy = args.gradient_surgery.lower() == "true"
    args.use_max_para = args.use_max_para.lower() == "true"
    args.pc_test = args.pc_test.lower() == "true"
    args.do_sentiment = args.do_sentiment.lower() == "true"
    args.always_save = args.always_save.lower() == "true"
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-{args.sampling}-multitask.pt'
    seed_everything(args.seed)  # Fix the seed for reproducibility.

    if args.train:
        train_multitask(args)
    else:
        args.filepath = args.load_filepath

    test_multitask(args)
