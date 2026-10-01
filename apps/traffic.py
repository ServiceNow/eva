import itertools
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ENDPOINT_LABELS = {
    "views": {"count": "Visitors", "uniques": "Unique visitors"},
    "clones": {"uniques": "Unique cloners"},
}


def parse_file_path(path: Path):
    collected_on, _, endpoint = path.stem.rpartition("_")
    return endpoint, date.fromisoformat(collected_on), path


def read_json(endpoint: str, path: Path):
    payload = json.loads(path.read_text())
    for entry in payload[endpoint]:
        day = datetime.fromisoformat(entry["timestamp"].removesuffix("Z")).date()
        yield day, entry


def load_series(endpoint: str, snapshots: list[tuple[date, Path]]) -> pd.DataFrame:
    # Snapshots can overlap (e.g. not fetched exactly 14 days apart). For any day covered by
    # multiple snapshots, keep the value from the most-recently-collected one.
    by_day: dict[date, tuple[date, dict]] = {}
    for collected_on, path in snapshots:
        for day, entry in read_json(endpoint, path):
            if day not in by_day or collected_on >= by_day[day][0]:
                by_day[day] = (collected_on, entry)

    labels = ENDPOINT_LABELS[endpoint]
    return pd.DataFrame.from_records(
        [
            {"Date": day, **{label: entry[field] for field, label in labels.items()}}
            for day, (_, entry) in sorted(by_day.items())
        ],
        index="Date",
    )


def main():
    st.set_page_config(page_title="EVA GitHub Traffic", page_icon="website/public/favicon.svg")
    st.header("EVA GitHub Traffic", text_alignment="center")

    entries = sorted(parse_file_path(path) for path in Path(".github/traffic").glob("*.json"))

    series = {
        endpoint: load_series(endpoint, [(collected_on, path) for _, collected_on, path in group])
        for endpoint, group in itertools.groupby(entries, key=lambda entry: entry[0])
    }
    df = pd.concat(series, axis="columns")
    df.columns = df.columns.droplevel(0)

    st.line_chart(df)
    st.caption("Historical data collected from https://github.com/ServiceNow/eva/graphs/traffic.")
    st.expander("Table", type="compact").dataframe(df)


if __name__ == "__main__":
    main()
