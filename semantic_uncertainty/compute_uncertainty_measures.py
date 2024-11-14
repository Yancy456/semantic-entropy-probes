"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
from copy import deepcopy
import logging
import os
import pickle
import random
import numpy as np
import wandb

from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils
from semantic.semantic_clustering import SemanticClustering

utils.setup_logger()

EXP_DETAILS = 'experiment_details.pkl'

def main(args):
    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"

    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=False, exist_ok=True, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored

    is_ood_eval = False  # pylint: disable=invalid-name
    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)

    if 'semantic_ids' not in result_dict:
        result_dict['semantic_ids'] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies, accuracies = defaultdict(list), defaultdict(list)
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # Loop over datapoints and compute validation embeddings, accuracies and entropies.
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example['question']
        full_responses = example["responses"]
        responses = [fr[0] for fr in full_responses]


        # Token log likelihoods. Shape = (n_sample, n_tokens)
        log_liks = [r[1] for r in full_responses]

        for i in log_liks:
            assert i

        if args.condition_on_question and args.entailment_model == 'deberta':
            responses = [f'{question} {r}' for r in responses]

        # Compute semantic ids.
        cluster=SemanticClustering(entailment_model)

        # Compute entropy from frequencies of cluster assignments.
        entropies['cluster_assignment_entropy'].append(cluster.get_entropy(responses,example))

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    utils.save(result_dict, 'uncertainty_measures.pkl')

if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
