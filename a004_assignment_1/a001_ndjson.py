import argparse
import time

from a004_assignment_1.a000_CFG import (
    RAW_DATA_FOLDER,
    NDJSON_FILE_NAME_TO_LOAD,
    SIZE,
    RANK,
    TEST_DATA_FOLDER,
    COMM,
    NDJSON_TOTAL_LINE_NUM,
    PIECES_DATA_FOLDER, FILE_PIECES_FOR_MPI_V4,
)
from a004_assignment_1.a002_utils import (
    write_data_to_ndjson,
    load_ndjson_file_multi_lines_to_list,
    aggregate_score_by_hour,
    split_list,
    join_dict_pieces_hour_score,
    load_ndjson_file_by_process,
    mpi_v3_subprocess,
    split_file,
    mpi_v4_subprocess,
    check_split_files_exist,
)
from a004_assignment_1.a003_top_k import high_level_api_sort_result


def mpi_v1():
    """Root process (rank 0) reads all data, then scatters chunks to worker processes."""
    if RANK == 0:
        records: list | None = load_ndjson_file_multi_lines_to_list(
            ndjson_path_for_loading=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
            use_filter=True,
        )
        print(f"1. rank={RANK}, Node finished reading data")

        records = split_list(lst=records, pieces_num=SIZE)
        print(f"2. rank={RANK}, Data splitting finished")
    else:
        records = None

    # Scatter the chunks from rank 0 to all processes (including rank 0 itself)
    received_msg = COMM.scatter(records, root=0)
    print(f"3. rank={RANK}, Data scattering finished")

    # Each process calculates scores for its received chunk
    hour_score = aggregate_score_by_hour(received_msg)
    print(f"4. rank={RANK}, Node statistics finished")

    # Gather the results from all processes back to rank 0
    all_hour_score = COMM.gather(hour_score, root=0)
    print(f"5. rank={RANK}, Gather finished")

    # Rank 0 merges the results and saves
    if RANK == 0:
        merged_score: dict = join_dict_pieces_hour_score(
            all_hour_score,
            value_type="scalar",
            mode="sum",
        )
        print(f"6. rank={RANK}, Aggregation finished")

        write_data_to_ndjson(
            records=merged_score,
            target_path=TEST_DATA_FOLDER / "gathered_v1.ndjson",  # Use specific name
            if_dict_is_single_dict=False,
        )
        print(f"7. rank={RANK}, Saving results to disk finished")


def mpi_v2():
    """All processes read their assigned chunk of the data file concurrently."""
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = NDJSON_TOTAL_LINE_NUM

    # Each process reads its portion of the file directly
    records = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=SIZE,
        r=RANK,
        use_filter=True,
    )
    print(f"1. rank={RANK}, Node finished reading data")

    # Each process calculates scores for its read chunk
    hour_score = aggregate_score_by_hour(records)
    print(f"2. rank={RANK}, Node finished individual statistics")

    # Gather results back to rank 0
    all_hour_score = COMM.gather(hour_score, root=0)
    print(f"3. rank={RANK}, Gather finished")

    # Rank 0 merges and saves
    if RANK == 0:
        merged_score = join_dict_pieces_hour_score(
            all_hour_score,
            value_type="scalar",
            mode="sum",
        )
        print(f"4. rank={RANK}, Aggregation on root node finished")

        write_data_to_ndjson(
            records=merged_score,
            target_path=TEST_DATA_FOLDER / "gathered_v2.ndjson",
            if_dict_is_single_dict=False,
        )
        print(f"5. rank={RANK}, Saving results to disk finished")


def mpi_v3():
    """All processes read data concurrently, calculating hourly and ID scores during the read process."""
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = NDJSON_TOTAL_LINE_NUM

    # Step 1: Each process reads its portion and calculates scores simultaneously
    # IMPORTANT: Assumes mpi_v3_subprocess now returns hour_score, id_score, failure_records
    hour_score, id_score, failure_records = mpi_v3_subprocess(
        input_ndjson_path=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=SIZE,
        r=RANK,
        use_filter=False,
    )
    print(f"Rank={RANK}, Node finished reading and statistics")

    # Step 2: Gather results from all processes to Rank 0
    all_hour_score = COMM.gather(hour_score, root=0)
    print(f"Rank={RANK}, Gather hour scores finished")
    all_id_scores = COMM.gather(id_score, root=0)
    print(f"Rank={RANK}, Gather ID scores finished")
    all_failure_records = COMM.gather(failure_records, root=0)
    print(f"Rank={RANK}, Gather failure records finished")

    # Step 3: Rank 0 merges results and saves
    if RANK == 0:
        merge_and_write_results(
            all_hour_score,
            all_id_scores,
            all_failure_records,
            "v3"
        )
        print(f"Rank=0: Saving failures to disk finished")


def mpi_v4():
    """
    Uses pre-split NDJSON files. Each process calculates scores using mpi_v4_subprocess.
    If SIZE == 1, runs sequentially mimicking the parallel aggregation pattern.
    If SIZE > 1, runs in parallel with results gathered and merged on rank 0.
    """
    if SIZE == 1:
        # ---Serial execution path (mimicking parallel aggregation)---
        print(f"--- Starting Serial Processing (Mimicking MPI Gather) of {FILE_PIECES_FOR_MPI_V4} Split Files ---")
        serial_start_time = time.time()

        all_hour_scores_serial = []
        all_id_scores_serial = []
        all_failed_records_serial = []  # list of lists

        try:
            base_name = NDJSON_FILE_NAME_TO_LOAD.rsplit(".", 1)[0]
            extension = NDJSON_FILE_NAME_TO_LOAD.rsplit(".", 1)[1]
            extension_with_dot = f".{extension}" if extension else ""
        except IndexError:
            base_name = NDJSON_FILE_NAME_TO_LOAD
            extension_with_dot = ""

        for i in range(FILE_PIECES_FOR_MPI_V4):
            split_file_name = f"{base_name}_piece_{i}{extension_with_dot}"
            split_file_path = PIECES_DATA_FOLDER / split_file_name

            if not split_file_path.is_file():
                # Handle missing file
                print(f"Warning: Split file piece missing, skipping: {split_file_path}")
                continue

            print(f"  Processing piece {i}: {split_file_path}...")
            hour_score_piece, id_score_piece, failed_records_piece = mpi_v4_subprocess(
                file_path=split_file_path,
                use_filter=False,
            )

            # Store results rather than merging immediately
            all_hour_scores_serial.append(hour_score_piece)
            all_id_scores_serial.append(id_score_piece)
            all_failed_records_serial.append(failed_records_piece)

            print(f"  Finished processing piece {i}.")

        print("Completed processing all pieces sequentially.")

        # Call the refactored merge and write function
        merge_and_write_results(
            list_of_hour_scores=all_hour_scores_serial,
            list_of_id_scores=all_id_scores_serial,
            list_of_failed_records=all_failed_records_serial,
            filename_suffix="v4_serial_mimic"  # suffix for the serial run
        )

        # Print total time for serial run
        serial_end_time = time.time()
        elapsed_time = serial_end_time - serial_start_time
        print(
            f"--- Total serial processing time (mimicking gather) for "
            f"{FILE_PIECES_FOR_MPI_V4} pieces: {elapsed_time:.5f} seconds ---"
        )

    else:
        # ---Parallel execution path (SIZE > 1)---
        try:
            base_name = NDJSON_FILE_NAME_TO_LOAD.rsplit(".", 1)[0]
            extension = NDJSON_FILE_NAME_TO_LOAD.rsplit(".", 1)[1]
            split_file_name = f"{base_name}_piece_{RANK}.{extension}"
        except IndexError:
            split_file_name = f"{NDJSON_FILE_NAME_TO_LOAD}_piece_{RANK}"
        split_file_path = PIECES_DATA_FOLDER / split_file_name

        # Call mpi_v4_subprocess
        hour_score, id_score, failed_records = mpi_v4_subprocess(
            file_path=split_file_path,
            use_filter=False,
        )

        # Print processing info
        print(
            f"Rank={RANK}: "
            f"Processed file {split_file_name}. Found {len(hour_score)} hour scores, "
            f"{len(id_score)} ID scores. {len(failed_records)} failures."
        )

        all_hour_scores = COMM.gather(hour_score, root=0)
        all_id_scores = COMM.gather(id_score, root=0)
        all_failed_records = COMM.gather(failed_records, root=0)
        print(f"Rank={RANK}: Gather finished.")

        if RANK == 0:
            # Call the refactored merge and write function
            merge_and_write_results(
                list_of_hour_scores=all_hour_scores,
                list_of_id_scores=all_id_scores,
                list_of_failed_records=all_failed_records,
                filename_suffix="v4",
            )


def merge_and_write_results(
        list_of_hour_scores: list,
        list_of_id_scores: list,
        list_of_failed_records: list,  # list of lists
        filename_suffix: str
):
    """
Combines aggregated results collected from each part/process and writes the final merged data to the output file.

Args:
    list_of_hour_scores: A list of hour_score dictionaries from each part or process.
    list_of_id_scores: A list of id_score dictionaries from each part or process.
    list_of_failed_records: A list of lists containing failed records from each part or process.
    filename_suffix: A string suffix appended to the base output file name (e.g. "", "_serial_mimic").
    """
    caller_prefix = "Rank=0" if RANK == 0 and SIZE > 1 else "Serial merge"

    print(f"{caller_prefix}: Starting final merge and write...")

    # 1. Merge hourly scores
    merged_hour_score: dict = join_dict_pieces_hour_score(
        list_of_hour_scores,
        value_type="scalar",
        mode="sum",
    )
    print(f"{caller_prefix}: Hourly score merge finished ({len(merged_hour_score)} keys)")

    # 2. Merge ID scores
    merged_id_score: dict = join_dict_pieces_hour_score(
        list_of_id_scores,
        value_type="list",
        mode="sum",
    )
    print(f"{caller_prefix}: ID score merge finished ({len(merged_id_score)} keys)")

    # 3. Collect all failure records (flatten list of lists)
    merged_failures: list = []
    if list_of_failed_records:
        for sub_failures in list_of_failed_records:
            if sub_failures:
                merged_failures.extend(sub_failures)
    print(f"{caller_prefix}: Collected {len(merged_failures)} failure records")

    # 4. Define output path using suffix
    output_hour_path = TEST_DATA_FOLDER / f"merged_hour_score_{filename_suffix}.ndjson"
    output_id_path = TEST_DATA_FOLDER / f"merged_id_score_{filename_suffix}.ndjson"
    output_failures_path = TEST_DATA_FOLDER / f"merged_failures_{filename_suffix}.ndjson"

    # Ensure output directory exists
    TEST_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    # 5. Write final results and failure records
    print(f"{caller_prefix}: Writing results with suffix '{filename_suffix}'...")
    write_data_to_ndjson(
        records=merged_hour_score,
        target_path=output_hour_path,
        if_dict_is_single_dict=False,
    )
    write_data_to_ndjson(
        records=merged_id_score,
        target_path=output_id_path,
        if_dict_is_single_dict=False,
    )
    write_data_to_ndjson(
        records=merged_failures,
        target_path=output_failures_path,
        if_dict_is_single_dict=None,
    )

    print(f"{caller_prefix}: Writing complete to {TEST_DATA_FOLDER}")


def measure_mpi(func):
    """Decorator or wrapper to measure execution time for MPI functions, executed by rank 0."""
    start_time = 0.0
    if RANK == 0:
        print(f"Starting measurement for {func.__name__}...")
        start_time = time.time()

    func()

    COMM.Barrier()

    if RANK == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' execution finished.")
        print(f"Total time consumption: {elapsed_time:.5f} seconds")


def try_split_file_by_rank0():
    """Function to call the file splitting utility (intended for Rank 0 execution)."""
    if RANK == 0:
        if not check_split_files_exist(
                original_file_path=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
                to_pieces_num=FILE_PIECES_FOR_MPI_V4,
                output_folder=PIECES_DATA_FOLDER,
        ):
            print("Rank=0: Starting file splitting...")
            split_file(
                file_path=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
                total_line_num=NDJSON_TOTAL_LINE_NUM,
                to_pieces_num=FILE_PIECES_FOR_MPI_V4,
                output_folder=PIECES_DATA_FOLDER,
                use_filter=True,
            )
            print("Rank=0: File splitting finished.")
        else:
            print("Rank=0: File splitting skipped, because the pieces already exist.")
    COMM.Barrier()


def get_args():
    parser = argparse.ArgumentParser(
        description="Run MPI processing with specified version."
    )
    parser.add_argument(
        '-v', '--version',
        type=int,
        choices=[3, 4],
        required=True,
        help='Specify the MPI version to run (3 or 4)'
    )
    return parser.parse_args()


def start_main():
    args = get_args()
    selected_version = args.version

    # Execute based on the selected version
    if selected_version == 3:
        if RANK == 0:
            print("--- Selected MPI v3 ---")
        measure_mpi(mpi_v3)
    elif selected_version == 4:
        if RANK == 0:
            print("--- Selected MPI v4 ---")
        try_split_file_by_rank0()
        measure_mpi(mpi_v4)
    else:
        # This branch theoretically won't run because choices=[3, 4] with required=True
        if RANK == 0:
            print(f"Error: Invalid version '{selected_version}' selected.")
            import sys
            sys.exit(1)

    # Subsequent steps
    time.sleep(0.1)
    COMM.Barrier()  # Ensure all MPI tasks complete

    if RANK == 0:
        print(f"Rank=0: MPI processing (v{selected_version}) finished. Starting result sorting...")
        high_level_api_sort_result()
        print("Rank=0: Main script execution finished.")


if __name__ == "__main__":
    start_main()
