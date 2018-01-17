from datetime import datetime

import testing.postgresql
from sqlalchemy import create_engine

from triage.component.architect import utils
from .utils import create_binary_outcome_events


def test_df_from_query():
    events = [
        (1, datetime(2011, 1, 1), True),
        (2, datetime(2011, 1, 1), False),
        (3, datetime(2011, 1, 1), True),
        (4, datetime(2011, 1, 1), False),
        (5, datetime(2011, 1, 1), False),
        (1, datetime(2011, 2, 1), False),
        (2, datetime(2011, 2, 1), False),
        (3, datetime(2011, 2, 1), True),
        (4, datetime(2011, 2, 1), False),
        (5, datetime(2011, 2, 1), False),
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_binary_outcome_events(engine, 'events', events)
        df = utils.df_from_query(engine, 'select * from events where entity_id = 2', ['entity_id'])
        assert df.to_dict('records') == [
            {'outcome_date': '2011-01-01', 'outcome': 'f'},
            {'outcome_date': '2011-02-01', 'outcome': 'f'},
        ]
