import pprint
from pathlib import Path

from mpi4py import MPI

from a004_assignment_1.a002_utils import (
    write_data_to_ndjson,
    load_ndjson_file,
    aggregate_score_by_hour,
    split_list, join_dict_pieces,

)

DATA_FOLDER = Path("a003_data")
RAW_DATA_FOLDER = DATA_FOLDER / "a001_raw"
FILTERED_DATA_FOLDER = DATA_FOLDER / "a002_filtered"
TEST_DATA_FOLDER = DATA_FOLDER / "a003_test"

NDJSON_FILE_NAME_TO_LOAD = r"mastodon-16m.ndjson"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def start():
    records: list = load_ndjson_file(
        ndjson_path_for_loading=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
        use_filter=True,
    )
    write_data_to_ndjson(
        records=records, target_path=FILTERED_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    )
    pprint.pprint(aggregate_score_by_hour(records))


def start_mpi():
    if rank == 0:
        records: list | None = load_ndjson_file(
            ndjson_path_for_loading=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
            use_filter=True,
        )

        records = split_list(lst=records, pieces_num=size)

        # received_msg = comm.scatter(records, root=0)
        # hour_score: dict = aggregate_score_by_hour(received_msg)

        # target_name = get_ndjson_name_by_rank(
        #     original_name=NDJSON_FILE_NAME_TO_LOAD,
        #     r=rank,
        # )
        # target_path = TEST_DATA_FOLDER / target_name
        # write_data_to_ndjson(
        #     records=hour_score,
        #     target_path=target_path
        # )

        # gathered_msg = comm.gather(hour_score, root=0)
    else:
        records = None
        # received_msg = comm.scatter(None, root=0)
        # hour_score = aggregate_score_by_hour(received_msg)
        #
        # target_name = get_ndjson_name_by_rank(
        #     original_name=NDJSON_FILE_NAME_TO_LOAD,
        #     r=rank,
        # )
        # target_path = TEST_DATA_FOLDER / target_name
        # write_data_to_ndjson(
        #     records=hour_score,
        #     target_path=target_path
        # )
        # _ = comm.gather(hour_score, root=0)

    received_msg = comm.scatter(records, root=0)
    hour_score = aggregate_score_by_hour(received_msg)
    all_hour_score = comm.gather(hour_score, root=0)

    if rank == 0:
        all_hour_score = join_dict_pieces(all_hour_score, mode="sum")
        write_data_to_ndjson(
            records=all_hour_score,
            target_path=TEST_DATA_FOLDER / "gathered.ndjson"
        )


if __name__ == "__main__":
    start_mpi()
