"""Predict with LLM on task."""
import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import torch
import openai
import wandb

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute


# utils.setup_logger()
# openai.api_key = os.getenv("OPENAI_API_KEY")  # Set up OpenAI API credentials.

def main(args):
    random.seed(args.random_seed)

    wandb.init(
        # entity=entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        # dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        # notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )

    metric = utils.get_metric(args.metric)



################### Load Dataset ################################################

    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(
        train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(
            validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

###################################################################################


################### Create Few-Shot prompt ################################################
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    logging.info('Prompt is: %s', prompt)

####################################################################################


################### Initialize model ################################################
    model = utils.init_model(args)

####################################################################################


    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(
                set(remaining_answerable) | set(unanswerable_indices))
        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(
            args.num_samples, len(dataset)))

        if args.num_samples > len(dataset):
            logging.warning(
                'Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {
                'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []

            # We sample 1 low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy.

            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            for i in range(num_generations):
                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else args.temperature

                predicted_answer, token_log_likelihoods, (embedding, emb_last_before_gen, emb_before_eos) = model.predict(
                    local_prompt, temperature, return_latent=True)

                # Last token embedding
                embedding = embedding.cpu() if embedding is not None else None
                emb_last_before_gen = emb_last_before_gen.cpu(
                ) if emb_last_before_gen is not None else None
                emb_before_eos = emb_before_eos.cpu() if emb_before_eos is not None else None

                if i == 0:
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'emb_last_tok_before_gen': emb_last_before_gen,
                        'emb_tok_before_eos': emb_before_eos,
                    }

                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example),
                    })
                else:
                    logging.info('high-t prediction '.ljust(15) +
                                 str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding))

            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        if dataset_split == 'validation':
            utils.save(results_dict, 'uncertainty_measures.pkl')

    logging.info('Run complete.')
    del model


if __name__ == '__main__':
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
