import logging
import traceback
from functools import partial
from multiprocessing import Pool

from sqlalchemy import create_engine

from triage.component.catwalk.utils import Batch
from triage.component.catwalk.storage import InMemoryModelStorageEngine

from triage.experiments import ExperimentBase


class MultiCoreExperiment(ExperimentBase):

    def __init__(self, n_processes=1, n_db_processes=1, *args, **kwargs):
        super(MultiCoreExperiment, self).__init__(*args, **kwargs)
        self.n_processes = n_processes
        self.n_db_processes = n_db_processes
        if kwargs['model_storage_class'] == InMemoryModelStorageEngine:
            raise ValueError('''
                InMemoryModelStorageEngine not compatible with MultiCoreExperiment
            ''')

    def catwalk(self):
        for split_num, split in enumerate(self.full_matrix_definitions):
            self.log_split(split_num, split)
            train_store = self.matrix_store(split['train_uuid'])
            logging.info('Checking out train matrix')
            if train_store.empty:
                logging.warning('''Train matrix for split %s was empty,
                no point in training this model. Skipping
                ''', split['train_uuid'])
                continue
            logging.info('Checking out train labels')
            if len(train_store.labels().unique()) == 1:
                logging.warning('''Train Matrix for split %s had only one
                unique value, no point in training this model. Skipping
                ''', split['train_uuid'])
                continue

            logging.info('Training models')
            trainer_tasks = self.trainer.generate_train_tasks(
                grid_config=self.config['grid_config'],
                misc_db_parameters=dict(
                    test=False,
                    model_comment=self.config.get('model_comment', None),
                ),
                matrix_store=train_store
            )
            partial_train_models = partial(
                train_model,
                trainer_factory=self.trainer_factory,
                db_connection_string=self.db_engine.url
            )
            logging.info(
                'Starting parallel training: %s tasks, %s processes',
                len(trainer_tasks),
                self.n_processes
            )
            model_ids = []
            for batch_model_ids in self.parallelize(
                partial_train_models,
                trainer_tasks,
                self.n_processes
            ):
                model_ids += batch_model_ids
            # remove Nones from model_ids so we don't try to predict/evaluate on skipped baselines
            model_ids = [model_id for model_id in model_ids if model_id is not None]
            logging.info('Done training models')

            for split_def, test_uuid in zip(
                split['test_matrices'],
                split['test_uuids']
            ):
                as_of_times = split_def['as_of_times']
                logging.info(
                    'Testing and scoring as_of_times min: %s max: %s num: %s',
                    min(as_of_times),
                    max(as_of_times),
                    len(as_of_times)
                )
                test_store = self.matrix_store(test_uuid)
                if test_store.empty:
                    logging.warning('''Test matrix for train uuid %s
                    was empty, no point in training this model. Skipping
                    ''', split['train_uuid'])
                    continue
                partial_test_and_evaluate = partial(
                    test_and_evaluate,
                    predictor_factory=self.predictor_factory,
                    evaluator_factory=self.evaluator_factory,
                    indiv_importance_factory=self.indiv_importance_factory,
                    test_store=test_store,
                    train_store=train_store,
                    db_connection_string=self.db_engine.url,
                    split_def=split_def,
                    config=self.config
                )
                logging.info(
                    'Starting parallel testing with %s processes',
                    self.n_db_processes
                )
                self.parallelize_with_success_count(
                    partial_test_and_evaluate,
                    model_ids,
                    self.n_db_processes
                )
                logging.info('Cleaned up concurrent pool')
            logging.info('Done with test matrix')
        logging.info('Done with split')

    def parallelize_with_success_count(
        self,
        partially_bound_function,
        tasks,
        n_processes,
        chunksize=1
    ):
        num_successes = 0
        num_failures = 0
        for successful in self.parallelize(
            partially_bound_function,
            tasks,
            n_processes,
            chunksize
        ):
            if successful:
                num_successes += 1
            else:
                num_failures += 1
        logging.info(
            'Done. successes: %s, failures: %s',
            num_successes,
            num_failures
        )

    def build_matrices(self):
        logging.info('Creating sparse states')
        self.generate_sparse_states()
        logging.info('Creating labels')
        self.generate_labels()
        logging.info(
            'Creating feature tables with %s processes',
            self.n_db_processes
        )
        for table_name, tasks in self.feature_aggregation_table_tasks.items():
            logging.info('Processing features for %s', table_name)
            self.feature_generator.run_commands(tasks.get('prepare', []))
            partial_insert = partial(
                insert_into_table,
                feature_generator_factory=self.feature_generator_factory,
                db_connection_string=self.db_engine.url
            )
            self.parallelize_with_success_count(
                partial_insert,
                tasks.get('inserts', []),
                n_processes=self.n_db_processes,
                chunksize=25
            )
            self.feature_generator.run_commands(tasks.get('finalize', []))
            logging.info('%s completed', table_name)

        logging.info('Creating feature imputation tables')
        self.feature_generator.process_table_tasks(self.feature_imputation_table_tasks)

        partial_build_matrix = partial(
            build_matrix,
            planner_factory=self.planner_factory,
            db_connection_string=self.db_engine.url
        )
        logging.info(
            'Starting parallel matrix building: %s matrices, %s processes',
            len(self.matrix_build_tasks.keys()),
            self.n_processes
        )
        self.parallelize_with_success_count(
            partial_build_matrix,
            self.matrix_build_tasks.values(),
            self.n_processes
        )

    def parallelize(
        self,
        partially_bound_function,
        tasks,
        n_processes,
        chunksize=1,
    ):
        with Pool(n_processes, maxtasksperchild=1) as pool:
            for result in pool.map(
                partially_bound_function,
                [list(task_batch) for task_batch in Batch(tasks, chunksize)]
            ):
                yield result


def insert_into_table(
    insert_statements,
    feature_generator_factory,
    db_connection_string
):
    try:
        logging.info('Beginning insert batch')
        db_engine = create_engine(db_connection_string)
        feature_generator = feature_generator_factory(db_engine)
        feature_generator.run_commands(insert_statements)
        return True
    except Exception:
        logging.error('Child error: %s', traceback.format_exc())
        return False


def build_matrix(
    build_tasks,
    planner_factory,
    db_connection_string,
):
    try:
        db_engine = create_engine(db_connection_string)
        planner = planner_factory(engine=db_engine)

        for i, build_task in enumerate(build_tasks):
            planner.build_matrix(**build_task)

        return True
    except Exception:
        logging.error('Child error: %s', traceback.format_exc())
        return False


def train_model(
    train_tasks,
    trainer_factory,
    db_connection_string,
):
    try:
        db_engine = create_engine(db_connection_string)
        trainer = trainer_factory(db_engine=db_engine)
        return [
            trainer.process_train_task(**train_task)
            for train_task in train_tasks
        ]
    except Exception:
        logging.error('Child error: %s', traceback.format_exc())
        return []


def test_and_evaluate(
    model_ids,
    predictor_factory,
    evaluator_factory,
    indiv_importance_factory,
    test_store,
    train_store,
    db_connection_string,
    split_def,
    config
):
    try:
        db_engine = create_engine(db_connection_string)
        for model_id in model_ids:
            logging.info('Generating predictions for model id %s', model_id)
            predictor = predictor_factory(db_engine=db_engine)
            evaluator = evaluator_factory(db_engine=db_engine)
            individual_importance = indiv_importance_factory(db_engine=db_engine)

            logging.info('Generating individual importances for model id %s', model_id)
            individual_importance.calculate_and_save_all_methods_and_dates(
                model_id,
                test_store
            )
            logging.info('Generating evaluations for model id %s', model_id)

            #Generate predictions for the testing data then training data
            for store in (test_store, train_store):
                predictions_proba = predictor.predict(
                    model_id,
                    store,
                    misc_db_parameters=dict(),
                    train_matrix_columns=train_store.columns()
                )

                evaluator.evaluate(
                    predictions_proba=predictions_proba,
                    matrix_store=store,
                    model_id=model_id,
                    # for evaluation range, using first to last as of time:
                    evaluation_start_time=split_def['first_as_of_time'],
                    evaluation_end_time=split_def['last_as_of_time'],
                    as_of_date_frequency=split_def['test_as_of_date_frequency']
                )
        return True

    except Exception:
        logging.error('Child error: %s', traceback.format_exc())
        return False
