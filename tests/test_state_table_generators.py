from architect.state_table_generators import StateTableGenerator
import testing.postgresql
from datetime import datetime
from sqlalchemy.engine import create_engine
from tests.utils import assert_index, create_dense_state_table, create_binary_outcome_events


def test_sparse_state_table_generator():
    input_data = [
        (5, 'permitted', datetime(2016, 1, 1), datetime(2016, 6, 1)),
        (6, 'permitted', datetime(2016, 2, 5), datetime(2016, 5, 5)),
        (1, 'injail', datetime(2014, 7, 7), datetime(2014, 7, 15)),
        (1, 'injail', datetime(2016, 3, 7), datetime(2016, 4, 2)),
    ]

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_dense_state_table(engine, 'states', input_data)

        table_generator = StateTableGenerator(
            engine,
            'exp_hash',
            dense_state_table='states'
        )
        as_of_dates = [
            datetime(2016, 1, 1),
            datetime(2016, 2, 1),
            datetime(2016, 3, 1),
            datetime(2016, 4, 1),
            datetime(2016, 5, 1),
            datetime(2016, 6, 1),
        ]
        table_generator.generate_sparse_table(as_of_dates)
        results = [row for row in engine.execute(
            'select entity_id, as_of_date, injail, permitted from {} order by entity_id, as_of_date'.format(
                table_generator.sparse_table_name
            ))]
        expected_output = [
            # entity_id, as_of_date, injail, permitted
            (1, datetime(2016, 4, 1), True, False),
            (5, datetime(2016, 1, 1), False, True),
            (5, datetime(2016, 2, 1), False, True),
            (5, datetime(2016, 3, 1), False, True),
            (5, datetime(2016, 4, 1), False, True),
            (5, datetime(2016, 5, 1), False, True),
            (6, datetime(2016, 3, 1), False, True),
            (6, datetime(2016, 4, 1), False, True),
            (6, datetime(2016, 5, 1), False, True),
        ]
        assert results == expected_output
        assert_index(engine, table_generator.sparse_table_name, 'entity_id')
        assert_index(engine, table_generator.sparse_table_name, 'as_of_date')


def test_sparse_table_generator_from_events():
    input_data = [
        (1, datetime(2016, 1, 1), True),
        (1, datetime(2016, 4, 1), False),
        (1, datetime(2016, 3, 1), True),
        (2, datetime(2016, 1, 1), False),
        (2, datetime(2016, 1, 1), True),
        (3, datetime(2016, 1, 1), True),
        (5, datetime(2016, 1, 1), True),
        (5, datetime(2016, 1, 1), True),
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_binary_outcome_events(engine, 'events', input_data)
        table_generator = StateTableGenerator(
            engine,
            'exp_hash',
            events_table='events'
        )
        as_of_dates = [
            datetime(2016, 1, 1),
            datetime(2016, 2, 1),
            datetime(2016, 3, 1),
            datetime(2016, 4, 1),
            datetime(2016, 5, 1),
            datetime(2016, 6, 1),
        ]
        table_generator.generate_sparse_table(as_of_dates)
        expected_output = [
            (1, datetime(2016, 1, 1), True),
            (1, datetime(2016, 2, 1), True),
            (1, datetime(2016, 3, 1), True),
            (1, datetime(2016, 4, 1), True),
            (1, datetime(2016, 5, 1), True),
            (1, datetime(2016, 6, 1), True),
            (2, datetime(2016, 1, 1), True),
            (2, datetime(2016, 2, 1), True),
            (2, datetime(2016, 3, 1), True),
            (2, datetime(2016, 4, 1), True),
            (2, datetime(2016, 5, 1), True),
            (2, datetime(2016, 6, 1), True),
            (3, datetime(2016, 1, 1), True),
            (3, datetime(2016, 2, 1), True),
            (3, datetime(2016, 3, 1), True),
            (3, datetime(2016, 4, 1), True),
            (3, datetime(2016, 5, 1), True),
            (3, datetime(2016, 6, 1), True),
            (5, datetime(2016, 1, 1), True),
            (5, datetime(2016, 2, 1), True),
            (5, datetime(2016, 3, 1), True),
            (5, datetime(2016, 4, 1), True),
            (5, datetime(2016, 5, 1), True),
            (5, datetime(2016, 6, 1), True),
        ]
        results = [row for row in engine.execute(
            '''
                select entity_id, as_of_date, active from {}
                order by entity_id, as_of_date
            '''.format(
                table_generator.sparse_table_name
            )
        )]
        assert results == expected_output
        assert_index(engine, table_generator.sparse_table_name, 'entity_id')
        assert_index(engine, table_generator.sparse_table_name, 'as_of_date')
