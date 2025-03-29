import pprint
from pathlib import Path

from mpi4py import MPI

from a004_assignment_1.a002_utils import (
    write_data_to_ndjson,
    load_ndjson_file,
    aggregate_score_by_hour,
    split_list,
    join_dict_pieces_hour_score,
    load_ndjson_file_by_process,

)

DATA_FOLDER = Path("a003_data")
RAW_DATA_FOLDER = DATA_FOLDER / "a001_raw"
FILTERED_DATA_FOLDER = DATA_FOLDER / "a002_filtered"
TEST_DATA_FOLDER = DATA_FOLDER / "a003_test"

NDJSON_FILE_NAME_TO_LOAD = r"mastodon-106k.ndjson"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def start_one_core():
    records: list = load_ndjson_file(
        ndjson_path_for_loading=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
        use_filter=True,
    )
    write_data_to_ndjson(
        records=records, target_path=FILTERED_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    )
    pprint.pprint(aggregate_score_by_hour(records))


def start_mpi_v1():
    """主进程读取全部数据，然后分发给工作进程"""
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
        all_hour_score = join_dict_pieces_hour_score(all_hour_score, mode="sum")
        write_data_to_ndjson(
            records=all_hour_score,
            target_path=TEST_DATA_FOLDER / "gathered.ndjson"
        )


def start_mpi_v2():
    """所有进程同时开始读取数据"""
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = 4500

    records = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=size,
        r=rank,
        use_filter=True,
    )

    hour_score = aggregate_score_by_hour(records)
    all_hour_score = comm.gather(hour_score, root=0)

    if rank == 0:
        all_hour_score = join_dict_pieces_hour_score(all_hour_score, mode="sum")
        write_data_to_ndjson(
            records=all_hour_score,
            target_path=TEST_DATA_FOLDER / "gathered_v2.ndjson"
        )


def tst_load_ndjson_file_by_process():
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = 30

    records_0 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=0,
        use_filter=True,
    )
    records_1 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=1,
        use_filter=True,
    )
    records_2 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=2,
        use_filter=True,
    )
    records_3 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=3,
        use_filter=True,
    )
    pass


if __name__ == "__main__":
    # start_mpi_v1()
    # tst_load_ndjson_file_by_process()
    start_mpi_v2()
