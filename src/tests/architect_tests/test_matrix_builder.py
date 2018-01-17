import csv
import datetime
import os
from unittest import TestCase
from collections import OrderedDict

import testing.postgresql
import pandas as pd
from mock import Mock
from sqlalchemy import create_engine

from triage.component import metta
from triage.component.architect.builders import MatrixBuilder
from triage.component.catwalk.storage import CSVMatrixStore, HDFMatrixStore

from .utils import (
    create_schemas,
    TemporaryDirectory,
    create_entity_date_df,
    convert_string_column_to_date
)


states = [
    [0, '2016-01-01', False, True],
    [0, '2016-02-01', False, True],
    [0, '2016-03-01', False, True],
    [0, '2016-04-01', False, True],
    [0, '2016-05-01', False, True],
    [1, '2016-01-01', True, False],
    [1, '2016-02-01', True, False],
    [1, '2016-03-01', True, False],
    [1, '2016-04-01', True, False],
    [1, '2016-05-01', True, False],
    [2, '2016-01-01', True, False],
    [2, '2016-02-01', True, True],
    [2, '2016-03-01', True, False],
    [2, '2016-04-01', True, True],
    [2, '2016-05-01', True, False],
    [3, '2016-01-01', False, True],
    [3, '2016-02-01', True, True],
    [3, '2016-03-01', False, True],
    [3, '2016-04-01', True, True],
    [3, '2016-05-01', False, True],
    [4, '2016-01-01', True, True],
    [4, '2016-02-01', True, True],
    [4, '2016-03-01', True, True],
    [4, '2016-04-01', True, True],
    [4, '2016-05-01', True, True],
    [5, '2016-01-01', False, False],
    [5, '2016-02-01', False, False],
    [5, '2016-03-01', False, False],
    [5, '2016-04-01', False, False],
    [5, '2016-05-01', False, False]
]

features0_pre = [
    [0, '2016-01-01', 2, 1],
    [1, '2016-01-01', 1, 2],
    [0, '2016-02-01', 2, 3],
    [1, '2016-02-01', 2, 4],
    [0, '2016-03-01', 3, 3],
    [1, '2016-03-01', 3, 4],
    [0, '2016-04-01', 4, 3],
    [1, '2016-05-01', 5, 4]
]

features1_pre = [
    [2, '2016-01-01', 1, 1],
    [3, '2016-01-01', 1, 2],
    [2, '2016-02-01', 2, 3],
    [3, '2016-02-01', 2, 2],
    [0, '2016-03-01', 3, 3],
    [1, '2016-03-01', 3, 4],
    [2, '2016-03-01', 3, 3],
    [3, '2016-03-01', 3, 4],
    [3, '2016-03-01', 3, 4],
    [0, '2016-03-01', 3, 3],
    [4, '2016-03-01', 1, 4],
    [5, '2016-03-01', 2, 4]
]

# collate will ensure every entity/date combination in the state
# table have an imputed value in the features table, so ensure
# this is true for our test (filling with 9's):
f0_dict = {(r[0], r[1]): r for r in features0_pre}
f1_dict = {(r[0], r[1]): r for r in features1_pre}

for rec in states:
    ent_dt = (rec[0], rec[1])
    f0_dict[ent_dt] = f0_dict.get(ent_dt, list(ent_dt + (9, 9)))
    f1_dict[ent_dt] = f1_dict.get(ent_dt, list(ent_dt + (9, 9)))

features0 = sorted(f0_dict.values(), key=lambda x: (x[1], x[0]))
features1 = sorted(f1_dict.values(), key=lambda x: (x[1], x[0]))

features_tables = [features0, features1]

# make some fake labels data

labels = [
    [0, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [0, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [1, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-03-01', '1 month', 'booking', 'binary', 1],
    [2, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [2, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-05-01', '1 month', 'ems',     'binary', 1],
    [3, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-03-01', '1 month', 'booking', 'binary', 1],
    [3, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [3, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-04-01', '1 month', 'ems',     'binary', 1],
    [3, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [4, '2016-01-01', '1 month', 'booking', 'binary', 1],
    [4, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [4, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [4, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [4, '2016-05-01', '1 month', 'booking', 'binary', 0],
    [4, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [4, '2016-02-01', '1 month', 'ems',     'binary', 1],
    [4, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [4, '2016-04-01', '1 month', 'ems',     'binary', 1],
    [4, '2016-05-01', '1 month', 'ems',     'binary', 1],
    [5, '2016-01-01', '1 month', 'booking', 'binary', 1],
    [5, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [5, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [5, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [5, '2016-05-01', '1 month', 'booking', 'binary', 0],
    [5, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [5, '2016-02-01', '1 month', 'ems',     'binary', 1],
    [5, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [5, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [5, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [0, '2016-03-01', '3 month', 'booking', 'binary', 0],
    [0, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [0, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [0, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [0, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [0, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [0, '2016-04-01', '3 month', 'ems',     'binary', 0],
    [0, '2016-05-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [1, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [1, '2016-03-01', '3 month', 'booking', 'binary', 0],
    [1, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [1, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [1, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-04-01', '3 month', 'ems',     'binary', 0],
    [1, '2016-05-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [2, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [2, '2016-03-01', '3 month', 'booking', 'binary', 1],
    [2, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [2, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [2, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-04-01', '3 month', 'ems',     'binary', 0],
    [2, '2016-05-01', '3 month', 'ems',     'binary', 1],
    [3, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [3, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [3, '2016-03-01', '3 month', 'booking', 'binary', 1],
    [3, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [3, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [3, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [3, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [3, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [3, '2016-04-01', '3 month', 'ems',     'binary', 1],
    [3, '2016-05-01', '3 month', 'ems',     'binary', 0],
    [3, '2016-05-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [4, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [4, '2016-03-01', '3 month', 'booking', 'binary', 1],
    [4, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [4, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [4, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-04-01', '3 month', 'ems',     'binary', 0],
    [4, '2016-05-01', '3 month', 'ems',     'binary', 1],
    [5, '2016-01-01', '3 month', 'booking', 'binary', 0],
    [5, '2016-02-01', '3 month', 'booking', 'binary', 0],
    [5, '2016-03-01', '3 month', 'booking', 'binary', 1],
    [5, '2016-04-01', '3 month', 'booking', 'binary', 0],
    [5, '2016-05-01', '3 month', 'booking', 'binary', 1],
    [5, '2016-01-01', '3 month', 'ems',     'binary', 0],
    [5, '2016-02-01', '3 month', 'ems',     'binary', 0],
    [5, '2016-03-01', '3 month', 'ems',     'binary', 0],
    [5, '2016-04-01', '3 month', 'ems',     'binary', 1],
    [5, '2016-05-01', '3 month', 'ems',     'binary', 0]
]

label_name = 'booking'
label_type = 'binary'

db_config = {
    'features_schema_name': 'features',
    'labels_schema_name': 'labels',
    'labels_table_name': 'labels',
    'sparse_state_table_name': 'staging.sparse_states',
}


def test_make_entity_date_table():
    """ Test that the make_entity_date_table function contains the correct
    values.
    """
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0),
             datetime.datetime(2016, 3, 1, 0, 0)]

    # make a dataframe of entity ids and dates to test against
    ids_dates = create_entity_date_df(
        labels=labels,
        states=states,
        as_of_dates=dates,
        state_one=True,
        state_two=True,
        label_name='booking',
        label_type='binary',
        label_timespan='1 month'
    )

    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine,
            features_tables=features_tables,
            labels=labels,
            states=states
        )

        with TemporaryDirectory() as temp_dir:
            builder = MatrixBuilder(
                db_config=db_config,
                matrix_store_constructor=CSVMatrixStore,
                matrix_directory=temp_dir,
                engine=engine
            )
            engine.execute(
                'CREATE TABLE features.tmp_entity_date (a int, b date);'
            )
            # call the function to test the creation of the table
            entity_date_table_name = builder.make_entity_date_table(
                as_of_times=dates,
                label_type='binary',
                label_name='booking',
                state='state_one AND state_two',
                matrix_uuid='my_uuid',
                matrix_type='train',
                label_timespan='1 month'
            )

            # read in the table
            result = pd.read_sql(
                "select * from features.{} order by entity_id, as_of_date".format(entity_date_table_name),
                engine
            )

            # compare the table to the test dataframe
            test = (result == ids_dates)
            assert(test.all().all())


def test_retrieve_features_data():
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0)]

    # make dataframe for entity ids and dates
    ids_dates = create_entity_date_df(
        labels=labels,
        states=states,
        as_of_dates=dates,
        state_one=True,
        state_two=True,
        label_name='booking',
        label_type='binary',
        label_timespan='1 month'
    )

    features = [['f1', 'f2'], ['f3', 'f4']]
    # make dataframes of features to test against
    expected_features_dfs = []
    for i, table in enumerate(features_tables):
        expected_index = ['entity_id', 'as_of_date']
        cols = expected_index + features[i]
        temp_df = pd.DataFrame(
            table,
            columns=cols
        )
        temp_df['as_of_date'] = convert_string_column_to_date(temp_df['as_of_date'])
        expected_features_dfs.append(
            ids_dates.merge(
                right=temp_df,
                how='left',
                on=expected_index
            ).set_index(expected_index)
        )

    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_schemas(
            engine=engine,
            features_tables=features_tables,
            labels=labels,
            states=states
        )

        builder = MatrixBuilder(
            db_config=db_config,
            matrix_store_constructor=CSVMatrixStore,
            matrix_directory='',
            engine=engine,
        )

        # make the entity-date table
        entity_date_table_name = builder.make_entity_date_table(
            as_of_times=dates,
            label_type='binary',
            label_name='booking',
            state='state_one AND state_two',
            matrix_type='train',
            matrix_uuid='my_uuid',
            label_timespan='1 month'
        )

        feature_dictionary = OrderedDict(
            ('features{}'.format(i), feature_list) for i, feature_list in enumerate(features)
        )

        features_dataframes = builder.retrieve_features_data(
            as_of_times=dates,
            feature_dictionary=feature_dictionary,
            entity_date_table_name=entity_date_table_name,
            matrix_uuid='my_uuid'
        )

        # get the queries and test them
        for result_df, expected_df in zip(features_dataframes, expected_features_dfs):
            test = (result_df == expected_df)
            assert(test.all().all())


def test_retrieve_labels_data():
    """ Test the write_labels_data function by checking whether the query
    produces the correct labels
    """
    # set up labeling config variables
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0)]


    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_schemas(
            engine,
            features_tables,
            labels,
            states
        )
        builder = MatrixBuilder(
            db_config=db_config,
            matrix_directory='',
            matrix_store_constructor=CSVMatrixStore,
            engine=engine,
        )

        # make the entity-date table
        entity_date_table_name = builder.make_entity_date_table(
            as_of_times=dates,
            label_type='binary',
            label_name='booking',
            state='state_one AND state_two',
            matrix_type='train',
            matrix_uuid='my_uuid',
            label_timespan='1 month'
        )

        result_df = builder.retrieve_labels_data(
            label_name=label_name,
            label_type=label_type,
            label_timespan='1 month',
            matrix_uuid='my_uuid',
            entity_date_table_name=entity_date_table_name,
        )
        expected_df = pd.DataFrame.from_dict({
            'entity_id': [2, 3, 4, 4],
            'as_of_date': ['2016-02-01', '2016-02-01', '2016-01-01', '2016-02-01'],
            'booking': [0, 0, 1, 0],
        })
        expected_df['as_of_date'] = convert_string_column_to_date(expected_df['as_of_date'])
        expected_df.set_index(['entity_id', 'as_of_date'], inplace=True)
        test = (result_df == expected_df)
        assert(test.all().all())


class TestBuildMatrix(TestCase):
    def test_train_matrix(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states
            )

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_directory=temp_dir,
                    engine=engine,
                    matrix_store_constructor=CSVMatrixStore
                )
                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f3', 'f4'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'state': 'state_one AND state_two',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'feature_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'label_timespan': '1 month'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                builder.build_matrix(
                    as_of_times=dates,
                    label_name='booking',
                    label_type='binary',
                    feature_dictionary=feature_dictionary,
                    matrix_directory=temp_dir,
                    matrix_metadata=matrix_metadata,
                    matrix_uuid=uuid,
                    matrix_type='train'
                )

                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )
                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 6)

    def test_test_matrix(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states
            )

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_directory=temp_dir,
                    engine=engine,
                    matrix_store_constructor=CSVMatrixStore
                )

                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f3', 'f4'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'state': 'state_one AND state_two',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'feature_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'label_timespan': '1 month'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                builder.build_matrix(
                    as_of_times=dates,
                    label_name='booking',
                    label_type='binary',
                    feature_dictionary=feature_dictionary,
                    matrix_directory=temp_dir,
                    matrix_metadata=matrix_metadata,
                    matrix_uuid=uuid,
                    matrix_type='test'
                )
                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )

                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 6)

    def test_nullcheck(self):
        f0_dict = {(r[0], r[1]): r for r in features0_pre}
        f1_dict = {(r[0], r[1]): r for r in features1_pre}

        features0 = sorted(f0_dict.values(), key=lambda x: (x[1], x[0]))
        features1 = sorted(f1_dict.values(), key=lambda x: (x[1], x[0]))

        features_tables = [features0, features1]

        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states
            )

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_directory=temp_dir,
                    engine=engine,
                    matrix_store_constructor=CSVMatrixStore
                )

                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f3', 'f4'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'state': 'state_one AND state_two',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'feature_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'label_timespan': '1 month'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                with self.assertRaises(ValueError):
                    builder.build_matrix(
                        as_of_times=dates,
                        label_name='booking',
                        label_type='binary',
                        feature_dictionary=feature_dictionary,
                        matrix_directory=temp_dir,
                        matrix_metadata=matrix_metadata,
                        matrix_uuid=uuid,
                        matrix_type='test'
                    )

    def test_replace(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_schemas(
                engine=engine,
                features_tables=features_tables,
                labels=labels,
                states=states
            )

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                builder = MatrixBuilder(
                    db_config=db_config,
                    matrix_directory=temp_dir,
                    engine=engine,
                    matrix_store_constructor=CSVMatrixStore,
                    replace=False
                )

                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f3', 'f4'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'state': 'state_one AND state_two',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'feature_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'label_timespan': '1 month'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                builder.build_matrix(
                    as_of_times=dates,
                    label_name='booking',
                    label_type='binary',
                    feature_dictionary=feature_dictionary,
                    matrix_directory=temp_dir,
                    matrix_metadata=matrix_metadata,
                    matrix_uuid=uuid,
                    matrix_type='test'
                )

                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )

                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 6)

                # rerun
                builder.make_entity_date_table = Mock()
                builder.build_matrix(
                    as_of_times=dates,
                    label_name='booking',
                    label_type='binary',
                    feature_dictionary=feature_dictionary,
                    matrix_directory=temp_dir,
                    matrix_metadata=matrix_metadata,
                    matrix_uuid=uuid,
                    matrix_type='test'
                )
                assert not builder.make_entity_date_table.called
