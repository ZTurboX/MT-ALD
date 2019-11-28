
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random


import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertModel, BertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from compute_score import compute_metrics


from utils import load_and_cache_examples,get_char_vocab,char2ids
from data_processor import AggressionProcessor,AttackProcessor,ToxicityProcessor,Multi_Task_Processor

from multi_task_model import Multi_Model

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                  (BertConfig,)), ())

MODEL_CLASSES = {'bert': (BertConfig, BertModel, BertTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



processors = {"aggression": AggressionProcessor,"attack":AttackProcessor,"toxicity":ToxicityProcessor,"multi_task":Multi_Task_Processor}
output_modes = {"aggression": "classification", "attack":"classification","toxicity":"classification","multi_task":"classification"}

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    aggression_tensor = torch.tensor(tokenizer.encode("aggression"), dtype=torch.long).to(args.device)
    attack_tensor = torch.tensor(tokenizer.encode("attack"), dtype=torch.long).to(args.device)
    toxicity_tensor = torch.tensor(tokenizer.encode("toxicity"), dtype=torch.long).to(args.device)

    char_vocab=get_char_vocab()
    aggression_char_ids=char2ids("aggression",char_vocab)
    attack_char_ids=char2ids("attack",char_vocab)
    toxicity_char_ids=char2ids("toxicity",char_vocab)

    aggression_char_tenor=torch.tensor(aggression_char_ids,dtype=torch.long).to(args.device)
    attack_char_tenor = torch.tensor(attack_char_ids, dtype=torch.long).to(args.device)
    toxicity_char_tenor = torch.tensor(toxicity_char_ids, dtype=torch.long).to(args.device)


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_f1=0.0
    best_aggression_score={}
    best_attack_score={}
    best_toxicity_score={}
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                      'aggression_labels': batch[3],'attack_labels': batch[4],'toxicity_labels': batch[5],
                      'aggression_tensor':aggression_tensor,'attack_tensor':attack_tensor,'toxicity_tensor':toxicity_tensor,
                      'aggression_char_tensor':aggression_char_tenor,'attack_char_tensor':attack_char_tenor,'toxicity_char_tensor':toxicity_char_tenor}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            if args.all_task:
                aggression_logits, attack_logits, toxicity_logits, loss = model(**inputs)
            if args.aggression_attack_task:
                aggression_logits, attack_logits, loss= model(**inputs)
            if args.aggression_toxicity_task:
                aggression_logits, toxicity_logits, loss=model(**inputs)
            if args.attack_toxicity_task:
                attack_logits, toxicity_logits, loss=model(**inputs)



            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 :
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics

                    if args.all_task:

                        aggression_results, attack_results, toxicity_results = evaluate(args, model, tokenizer)

                        aggression_f1=aggression_results['score']['f1']
                        attack_f1=attack_results['score']['f1']
                        toxicity_f1=toxicity_results['score']['f1']

                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

                        if (aggression_f1+attack_f1+toxicity_f1)/3.0>best_f1:
                            best_f1=(aggression_f1+attack_f1+toxicity_f1)/3.0
                            best_aggression_score.update(aggression_results)
                            best_attack_score.update(attack_results)
                            best_toxicity_score.update(toxicity_results)

                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            output_model_file = os.path.join(output_dir, "model.pt")
                            torch.save(model.state_dict(), output_model_file)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            aggression_output_eval_file = os.path.join(args.output_dir, "aggression_eval_results.txt")
                            attack_output_eval_file = os.path.join(args.output_dir, "attack_eval_results.txt")
                            toxicity_output_eval_file = os.path.join(args.output_dir, "toxicity_eval_results.txt")
                            with open(aggression_output_eval_file, "a") as writer:

                                for key in sorted(aggression_results.keys()):

                                    writer.write("checkpoint%s-%s = %s\n" % (str(global_step),key, str(aggression_results[key])))

                            with open(attack_output_eval_file, "a") as writer:

                                for key in sorted(attack_results.keys()):

                                    writer.write("checkpoint%s-%s = %s\n" % (str(global_step),key, str(attack_results[key])))

                            with open(toxicity_output_eval_file, "a") as writer:

                                for key in sorted(toxicity_results.keys()):

                                    writer.write("checkpoint%s-%s = %s\n" % (str(global_step),key, str(toxicity_results[key])))

                        logger.info("************* best  results ***************")
                        for key in sorted(best_aggression_score.keys()):
                            logger.info("aggression-%s = %s", key, str(best_aggression_score[key]))

                        for key in sorted(best_attack_score.keys()):
                            logger.info("attack-%s = %s", key, str(best_attack_score[key]))

                        for key in sorted(best_toxicity_score.keys()):
                            logger.info("toxicity-%s = %s", key, str(best_toxicity_score[key]))

                    if args.aggression_attack_task:
                        aggression_results, attack_results = evaluate(args, model, tokenizer)

                        aggression_f1 = aggression_results['score']['f1']
                        attack_f1 = attack_results['score']['f1']


                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

                        if (aggression_f1 + attack_f1 ) / 2.0 > best_f1:
                            best_f1 = (aggression_f1 + attack_f1 ) / 2.0
                            best_aggression_score.update(aggression_results)
                            best_attack_score.update(attack_results)

                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            output_model_file = os.path.join(output_dir, "model.pt")
                            torch.save(model.state_dict(), output_model_file)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            aggression_output_eval_file = os.path.join(args.output_dir, "aggression_eval_results.txt")
                            attack_output_eval_file = os.path.join(args.output_dir, "attack_eval_results.txt")
                            with open(aggression_output_eval_file, "a") as writer:

                                for key in sorted(aggression_results.keys()):
                                    writer.write("checkpoint%s-%s = %s\n" % (
                                    str(global_step), key, str(aggression_results[key])))

                            with open(attack_output_eval_file, "a") as writer:

                                for key in sorted(attack_results.keys()):
                                    writer.write(
                                        "checkpoint%s-%s = %s\n" % (str(global_step), key, str(attack_results[key])))



                        logger.info("************* best  results ***************")
                        for key in sorted(best_aggression_score.keys()):
                            logger.info("aggression-%s = %s", key, str(best_aggression_score[key]))

                        for key in sorted(best_attack_score.keys()):
                            logger.info("attack-%s = %s", key, str(best_attack_score[key]))

                    if args.aggression_toxicity_task:
                        aggression_results, toxicity_results = evaluate(args, model, tokenizer)

                        aggression_f1 = aggression_results['score']['f1']

                        toxicity_f1 = toxicity_results['score']['f1']

                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

                        if (aggression_f1  + toxicity_f1) / 2.0 > best_f1:
                            best_f1 = (aggression_f1  + toxicity_f1) / 2.0
                            best_aggression_score.update(aggression_results)

                            best_toxicity_score.update(toxicity_results)

                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            output_model_file = os.path.join(output_dir, "model.pt")
                            torch.save(model.state_dict(), output_model_file)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            aggression_output_eval_file = os.path.join(args.output_dir, "aggression_eval_results.txt")

                            toxicity_output_eval_file = os.path.join(args.output_dir, "toxicity_eval_results.txt")
                            with open(aggression_output_eval_file, "a") as writer:

                                for key in sorted(aggression_results.keys()):
                                    writer.write("checkpoint%s-%s = %s\n" % (
                                    str(global_step), key, str(aggression_results[key])))


                            with open(toxicity_output_eval_file, "a") as writer:

                                for key in sorted(toxicity_results.keys()):
                                    writer.write(
                                        "checkpoint%s-%s = %s\n" % (str(global_step), key, str(toxicity_results[key])))

                        logger.info("************* best  results ***************")
                        for key in sorted(best_aggression_score.keys()):
                            logger.info("aggression-%s = %s", key, str(best_aggression_score[key]))



                        for key in sorted(best_toxicity_score.keys()):
                            logger.info("toxicity-%s = %s", key, str(best_toxicity_score[key]))

                    if args.attack_toxicity_task:
                        attack_results, toxicity_results = evaluate(args, model, tokenizer)


                        attack_f1 = attack_results['score']['f1']
                        toxicity_f1 = toxicity_results['score']['f1']

                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

                        if ( attack_f1 + toxicity_f1) / 2.0 > best_f1:
                            best_f1 = ( attack_f1 + toxicity_f1) / 2.0

                            best_attack_score.update(attack_results)
                            best_toxicity_score.update(toxicity_results)

                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            output_model_file = os.path.join(output_dir, "model.pt")
                            torch.save(model.state_dict(), output_model_file)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info("Saving model checkpoint to %s", output_dir)


                            attack_output_eval_file = os.path.join(args.output_dir, "attack_eval_results.txt")
                            toxicity_output_eval_file = os.path.join(args.output_dir, "toxicity_eval_results.txt")


                            with open(attack_output_eval_file, "a") as writer:

                                for key in sorted(attack_results.keys()):
                                    writer.write(
                                        "checkpoint%s-%s = %s\n" % (str(global_step), key, str(attack_results[key])))

                            with open(toxicity_output_eval_file, "a") as writer:

                                for key in sorted(toxicity_results.keys()):
                                    writer.write(
                                        "checkpoint%s-%s = %s\n" % (str(global_step), key, str(toxicity_results[key])))

                        logger.info("************* best  results ***************")
                        for key in sorted(best_aggression_score.keys()):
                            logger.info("aggression-%s = %s", key, str(best_aggression_score[key]))

                        for key in sorted(best_attack_score.keys()):
                            logger.info("attack-%s = %s", key, str(best_attack_score[key]))

                        for key in sorted(best_toxicity_score.keys()):
                            logger.info("toxicity-%s = %s", key, str(best_toxicity_score[key]))



            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    aggression_results = {}
    attack_results={}
    toxicity_results={}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        aggression_tensor = torch.tensor(tokenizer.encode("aggression"), dtype=torch.long).to(args.device)
        attack_tensor = torch.tensor(tokenizer.encode("attack"), dtype=torch.long).to(args.device)
        toxicity_tensor = torch.tensor(tokenizer.encode("toxicity"), dtype=torch.long).to(args.device)

        char_vocab = get_char_vocab()
        aggression_char_ids = char2ids("aggression", char_vocab)
        attack_char_ids = char2ids("attack", char_vocab)
        toxicity_char_ids = char2ids("toxicity", char_vocab)

        aggression_char_tenor = torch.tensor(aggression_char_ids, dtype=torch.long).to(args.device)
        attack_char_tenor = torch.tensor(attack_char_ids, dtype=torch.long).to(args.device)
        toxicity_char_tenor = torch.tensor(toxicity_char_ids, dtype=torch.long).to(args.device)


        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        aggression_preds = None
        attack_preds=None
        toxicity_preds=None
        aggression_out_label_ids = None
        attack_out_label_ids=None
        toxicity_out_label_ids=None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'aggression_labels': batch[3],
                          'attack_labels': batch[4], 'toxicity_labels': batch[5],
                          'aggression_tensor': aggression_tensor, 'attack_tensor': attack_tensor,
                          'toxicity_tensor': toxicity_tensor, 'aggression_char_tensor': aggression_char_tenor,
                          'attack_char_tensor': attack_char_tenor, 'toxicity_char_tensor': toxicity_char_tenor}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                               'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                if args.all_task:
                    aggression_logits, attack_logits, toxicity_logits, tmp_eval_loss = model(**inputs)
                if args.aggression_attack_task:
                    aggression_logits, attack_logits, tmp_eval_loss = model(**inputs)
                if args.aggression_toxicity_task:
                    aggression_logits, toxicity_logits, tmp_eval_loss = model(**inputs)
                if args.attack_toxicity_task:
                    attack_logits, toxicity_logits, tmp_eval_loss = model(**inputs)


                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if aggression_preds is None and attack_preds is None and toxicity_preds is None:
                if args.all_task:
                    aggression_preds = aggression_logits.detach().cpu().numpy()
                    aggression_out_label_ids = inputs['aggression_labels'].detach().cpu().numpy()

                    attack_preds = attack_logits.detach().cpu().numpy()
                    attack_out_label_ids = inputs['attack_labels'].detach().cpu().numpy()

                    toxicity_preds = toxicity_logits.detach().cpu().numpy()
                    toxicity_out_label_ids = inputs['toxicity_labels'].detach().cpu().numpy()
                if args.aggression_attack_task:
                    aggression_preds = aggression_logits.detach().cpu().numpy()
                    aggression_out_label_ids = inputs['aggression_labels'].detach().cpu().numpy()

                    attack_preds = attack_logits.detach().cpu().numpy()
                    attack_out_label_ids = inputs['attack_labels'].detach().cpu().numpy()
                if args.aggression_toxicity_task:
                    aggression_preds = aggression_logits.detach().cpu().numpy()
                    aggression_out_label_ids = inputs['aggression_labels'].detach().cpu().numpy()

                    toxicity_preds = toxicity_logits.detach().cpu().numpy()
                    toxicity_out_label_ids = inputs['toxicity_labels'].detach().cpu().numpy()
                if args.attack_toxicity_task:
                    attack_preds = attack_logits.detach().cpu().numpy()
                    attack_out_label_ids = inputs['attack_labels'].detach().cpu().numpy()

                    toxicity_preds = toxicity_logits.detach().cpu().numpy()
                    toxicity_out_label_ids = inputs['toxicity_labels'].detach().cpu().numpy()

            else:
                if args.all_task:
                    aggression_preds = np.append(aggression_preds, aggression_logits.detach().cpu().numpy(), axis=0)
                    aggression_out_label_ids = np.append(aggression_out_label_ids, inputs['aggression_labels'].detach().cpu().numpy(), axis=0)

                    attack_preds = np.append(attack_preds, attack_logits.detach().cpu().numpy(), axis=0)
                    attack_out_label_ids = np.append(attack_out_label_ids,inputs['attack_labels'].detach().cpu().numpy(), axis=0)

                    toxicity_preds = np.append(toxicity_preds, toxicity_logits.detach().cpu().numpy(), axis=0)
                    toxicity_out_label_ids = np.append(toxicity_out_label_ids,inputs['toxicity_labels'].detach().cpu().numpy(), axis=0)
                if args.aggression_attack_task:
                    aggression_preds = np.append(aggression_preds, aggression_logits.detach().cpu().numpy(), axis=0)
                    aggression_out_label_ids = np.append(aggression_out_label_ids,inputs['aggression_labels'].detach().cpu().numpy(), axis=0)

                    attack_preds = np.append(attack_preds, attack_logits.detach().cpu().numpy(), axis=0)
                    attack_out_label_ids = np.append(attack_out_label_ids,inputs['attack_labels'].detach().cpu().numpy(), axis=0)
                if args.aggression_toxicity_task:
                    aggression_preds = np.append(aggression_preds, aggression_logits.detach().cpu().numpy(), axis=0)
                    aggression_out_label_ids = np.append(aggression_out_label_ids,inputs['aggression_labels'].detach().cpu().numpy(), axis=0)
                    toxicity_preds = np.append(toxicity_preds, toxicity_logits.detach().cpu().numpy(), axis=0)
                    toxicity_out_label_ids = np.append(toxicity_out_label_ids,inputs['toxicity_labels'].detach().cpu().numpy(), axis=0)
                if args.attack_toxicity_task:
                    attack_preds = np.append(attack_preds, attack_logits.detach().cpu().numpy(), axis=0)
                    attack_out_label_ids = np.append(attack_out_label_ids,inputs['attack_labels'].detach().cpu().numpy(), axis=0)

                    toxicity_preds = np.append(toxicity_preds, toxicity_logits.detach().cpu().numpy(), axis=0)
                    toxicity_out_label_ids = np.append(toxicity_out_label_ids,inputs['toxicity_labels'].detach().cpu().numpy(), axis=0)





        eval_loss = eval_loss / nb_eval_steps

        if args.all_task:

            aggression_preds = np.argmax(aggression_preds, axis=1)
            attack_preds = np.argmax(attack_preds, axis=1)
            toxicity_preds = np.argmax(toxicity_preds, axis=1)


            aggression_result = compute_metrics(eval_task, aggression_preds, aggression_out_label_ids)
            aggression_results.update(aggression_result)

            attack_result = compute_metrics(eval_task, attack_preds, attack_out_label_ids)
            attack_results.update(attack_result)

            toxicity_result = compute_metrics(eval_task, toxicity_preds, toxicity_out_label_ids)
            toxicity_results.update(toxicity_result)

            logger.info("***** Eval aggression results {} *****".format(prefix))
            for key in sorted(aggression_result.keys()):
                logger.info("  %s = %s", key, str(aggression_result[key]))

            logger.info("***** Eval attack results {} *****".format(prefix))
            for key in sorted(attack_result.keys()):
                logger.info("  %s = %s", key, str(attack_result[key]))

            logger.info("***** Eval toxicity results {} *****".format(prefix))
            for key in sorted(toxicity_result.keys()):
                logger.info("  %s = %s", key, str(toxicity_result[key]))

        if args.aggression_attack_task:
            aggression_preds = np.argmax(aggression_preds, axis=1)
            attack_preds = np.argmax(attack_preds, axis=1)


            aggression_result = compute_metrics(eval_task, aggression_preds, aggression_out_label_ids)
            aggression_results.update(aggression_result)

            attack_result = compute_metrics(eval_task, attack_preds, attack_out_label_ids)
            attack_results.update(attack_result)



            logger.info("***** Eval aggression results {} *****".format(prefix))
            for key in sorted(aggression_result.keys()):
                logger.info("  %s = %s", key, str(aggression_result[key]))

            logger.info("***** Eval attack results {} *****".format(prefix))
            for key in sorted(attack_result.keys()):
                logger.info("  %s = %s", key, str(attack_result[key]))

        if args.aggression_toxicity_task:
            aggression_preds = np.argmax(aggression_preds, axis=1)

            toxicity_preds = np.argmax(toxicity_preds, axis=1)

            aggression_result = compute_metrics(eval_task, aggression_preds, aggression_out_label_ids)
            aggression_results.update(aggression_result)


            toxicity_result = compute_metrics(eval_task, toxicity_preds, toxicity_out_label_ids)
            toxicity_results.update(toxicity_result)

            logger.info("***** Eval aggression results {} *****".format(prefix))
            for key in sorted(aggression_result.keys()):
                logger.info("  %s = %s", key, str(aggression_result[key]))



            logger.info("***** Eval toxicity results {} *****".format(prefix))
            for key in sorted(toxicity_result.keys()):
                logger.info("  %s = %s", key, str(toxicity_result[key]))
        if args.attack_toxicity_task:

            attack_preds = np.argmax(attack_preds, axis=1)
            toxicity_preds = np.argmax(toxicity_preds, axis=1)


            attack_result = compute_metrics(eval_task, attack_preds, attack_out_label_ids)
            attack_results.update(attack_result)

            toxicity_result = compute_metrics(eval_task, toxicity_preds, toxicity_out_label_ids)
            toxicity_results.update(toxicity_result)


            logger.info("***** Eval attack results {} *****".format(prefix))
            for key in sorted(attack_result.keys()):
                logger.info("  %s = %s", key, str(attack_result[key]))

            logger.info("***** Eval toxicity results {} *****".format(prefix))
            for key in sorted(toxicity_result.keys()):
                logger.info("  %s = %s", key, str(toxicity_result[key]))


    if args.all_task:
        return aggression_results,attack_results,toxicity_results
    if args.aggression_attack_task:
        return aggression_results, attack_results
    if args.aggression_toxicity_task:
        return aggression_results, toxicity_results
    if args.attack_toxicity_task:
        return attack_results, toxicity_results




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="./data", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="./bert-base-uncased/bert-base-uncased-pytorch_model.bin", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))

    parser.add_argument("--task_name", default="multi_task", type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default='./check_points', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="./bert-base-uncased/bert-base-uncased-config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="./bert-base-uncased/bert-base-uncased-vocab.txt", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", default=True,action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--evaluate_during_training", default=True,action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=True,action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")


    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--mPos", default=1.5, type=float)
    parser.add_argument("--mNeg", default=1.5, type=float)
    parser.add_argument("--gamma", default=0.05, type=float)

    parser.add_argument("--all_task",default=False)
    parser.add_argument("--aggression_attack_task",default=False)
    parser.add_argument("--aggression_toxicity_task",default=False)
    parser.add_argument("--attack_toxicity_task",default=True)

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))


    # Setup CUDA, GPU & distributed training

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda:1", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)


    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    bert_config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
   
    bert_model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=bert_config)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.bert_vocab,
                                                do_lower_case=args.do_lower_case)

    model=Multi_Model(args,bert_config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



    # test
    results = {}
    if args.do_test and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

        logger.info("test the model")
        checkpoint='./check_points/checkpoint-200/model.pt'
        prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
        model.load_state_dict(torch.load(checkpoint))
        model.to(args.device)
        aggression_results, attack_results, toxicity_results = evaluate(args, model, tokenizer, prefix=prefix)









if __name__ == "__main__":
    main()
