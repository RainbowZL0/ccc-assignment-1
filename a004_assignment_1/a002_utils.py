import copy
import functools
import json
import pprint
import traceback
from datetime import datetime
from math import ceil
from pathlib import Path

from mpi4py import MPI


def load_ndjson_file_multi_lines_to_list(
        ndjson_path_for_loading: str | Path,
        use_filter: bool = False,
) -> list:
    """Reads data from an NDJSON file.

        Args:
            ndjson_path_for_loading (str | Path): Path to the NDJSON file.
            use_filter (bool, optional): Whether to filter each record. Defaults to False.

        Returns:
            list: A list of records read from the file.
    """
    records: list = []
    with open(ndjson_path_for_loading, "r", encoding="utf-8") as f:
        for line in f:
            record = parse_one_line(line, use_filter=use_filter)
            records.append(record)
    return records


def parse_one_line(line, use_filter):
    """Parses a single line from an NDJSON file."""
    if not line:
        return None
    line = line.strip()
    record = json.loads(line)
    if use_filter:
        record = filter_a_record(record)
    return record


def dict_to_a_line(dic):
    """Converts a dictionary to a JSON string followed by a newline."""
    line = json.dumps(dic, ensure_ascii=False)
    return f"{line}\n"


def filter_a_record(record: dict):
    """Filters a single record to extract key information.

    Args:
        record (dict): The original record dictionary.

    Returns:
        dict: The filtered record, containing only specified fields.
    """
    doc = record.get("doc", {})
    account = doc.get("account", {})
    return {
        "doc": {
            "createdAt": doc.get("createdAt"),
            "sentiment": doc.get("sentiment"),
            "account": {
                key: account.get(key)
                for key in (
                    "id",
                    "username",
                    # "acct",
                    # "uri",
                    # "url",
                    # "displayName",
                )
            },
        }
    }


def write_data_to_ndjson(
        records: list | dict,
        target_path: str | Path,
        if_dict_is_single_dict: bool | None,  # Added type hint based on usage
):
    """Writes data to an NDJSON file.

    Args:
        records (list | dict): The record(s) to write (single or multiple).
        target_path (str | Path): The target file path.
        if_dict_is_single_dict (bool | None): Specifies if a dict input represents a single record
                                               or multiple key-value pairs to be written line by line.
                                               Set to None if `records` is a list.
    """
    with open(target_path, "w", encoding="utf-8") as f:
        if isinstance(records, dict):
            if if_dict_is_single_dict:
                f.write(
                    dict_to_a_line(records)
                )
            else:
                # Assumes dict contains multiple records as key-value pairs
                for k, v in records.items():
                    f.write(
                        dict_to_a_line(
                            {k: v}
                        )
                    )
        elif isinstance(records, list):
            for record in records:
                f.write(
                    dict_to_a_line(record)
                )
        else:
            raise NotImplementedError(
                f"records must be a list or dict, got {type(records)}"
            )


def high_level_api_to_convert_raw_time_to_preferred_str(t):
    """Converts a raw time string to the 'YYYY-MM-DD HH' format string.

    Args:
        t (str): The original ISO 8601 time string.

    Returns:
        str: The formatted time string (down to the hour).
    """
    return iso_format_time_to_str(
        floor_time_to_the_latest_hour(raw_time_to_py_datetime(t))
    )


def raw_time_to_py_datetime(t):
    """Converts an ISO format time string to a Python datetime object.

    Args:
        t (str): The original time string, potentially ending with 'Z'.

    Returns:
        datetime: The Python datetime object.
    """
    # Handles ISO 8601 format, including the 'Z' UTC designator
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def floor_time_to_the_latest_hour(t):
    """Rounds down the time to the nearest whole hour.

    Args:
        t (datetime): The original datetime object.

    Returns:
        datetime: The datetime rounded down to the hour.
    """
    return t.replace(minute=0, second=0, microsecond=0)


def iso_format_time_to_str(t):
    """Formats a datetime object into a string ('YYYY-MM-DD HH:MM').

    Args:
        t (datetime): The datetime object.

    Returns:
        str: The formatted time string.
    """
    # Note: Original function name mentions 'HH:MM', but implementation uses 'HH' via floor_time.
    # Keeping '%Y-%m-%d %H:%M' as per original code, but effectively it will always be HH:00.
    # If you strictly want 'YYYY-MM-DD HH', use "%Y-%m-%d %H"
    return t.strftime("%Y-%m-%d %H:%M")


def aggregate_score_by_hour(records: list) -> dict:
    """Aggregates sentiment scores by the hour.

    Args:
        records (list): A list of records containing time and sentiment scores.

    Returns:
        dict: A dictionary where keys are hours (string format 'YYYY-MM-DD HH:00')
              and values are the total sentiment scores for that hour.
    """
    time_s_score: dict = {}
    for record in records:
        # Extract and format the creation time to the nearest hour
        created_hour: str = high_level_api_to_convert_raw_time_to_preferred_str(
            record["doc"]["createdAt"]
        )
        sentiment_score = record["doc"]["sentiment"]
        # print(created_hour) # Example: 2023-10-01 15:00

        # Initialize the score for the hour if it doesn't exist
        if created_hour not in time_s_score:
            time_s_score[created_hour] = 0.0
        # Add the current record's sentiment score to the hourly total
        time_s_score[created_hour] += sentiment_score
    return time_s_score


def split_list(lst, pieces_num):
    """Splits a list into a specified number of sublists.

    If the list cannot be evenly divided, the last sublist may contain fewer elements.

    Args:
        lst (list): The original list.
        pieces_num (int): The desired number of sublists.

    Returns:
        list: A list containing the generated sublists.
    """
    result = []
    elem_per_piece = ceil(len(lst) / pieces_num)
    for i in range(pieces_num):
        start = i * elem_per_piece
        end = min(start + elem_per_piece, len(lst))
        result.append(lst[start:end])
    return result


def join_dict_pieces_hour_score(lst, value_type, mode="sum"):
    """Merges multiple dictionaries, aggregating values by key.

    Args:
        value_type (str): The type of the dictionary values, either "scalar" or "list".
                          If "list", assumes the value is a list where aggregation happens on the first element.
        lst (list): A list of dictionaries to merge.
        mode (str, optional): The aggregation method. Currently, supports "sum". Defaults to "sum".

    Returns:
        dict: The merged dictionary.

    Raises:
        NotImplementedError: If value_type is not 'scalar' or 'list'.
        ValueError: If mode is not 'sum'.
    """
    if value_type not in ["scalar", "list"]:
        raise NotImplementedError(
            f"value_type must be scalar or list, but got {value_type}"
        )
    if mode != "sum":
        raise ValueError(f"mode '{mode}' not supported. Only 'sum' is implemented.")

    result = {}
    for dic in lst:
        for k, v in dic.items():
            if k not in result:
                # Use deep copy for lists to avoid modifying the original sub-dictionaries if they are reused
                result[k] = copy.deepcopy(v) if value_type == "list" else v
            else:
                if mode == "sum":
                    if value_type == "scalar":
                        result[k] += v
                    elif isinstance(result[k], list) and isinstance(v, list) and result[k] and v:
                        # Ensure both are non-empty lists before accessing the first element
                        result[k][0] += v[0]
                    else:
                        # Handle cases where one value might not be a list or is empty,
                        # depending on expected behavior (e.g., raise error, skip, default)
                        # Currently assumes valid list structure based on 'value_type' check
                        pass  # Or raise an error: raise TypeError(f"Incompatible types for summation on key {k}")

    return result


def load_ndjson_file_by_process(
        ndjson_path_for_loading,
        ndjson_line_num,
        process_num,
        r,  # Assuming 'r' is the rank of the current process (0-based)
        use_filter=False,
):
    """Loads a specific chunk of an NDJSON file based on process rank."""
    num_line_per_process = ceil(ndjson_line_num / process_num)
    # Calculate the line range [start, end) for this process (1-based indexing for lines)
    start_line = r * num_line_per_process + 1
    # The end line index is exclusive
    end_line = min(start_line + num_line_per_process, ndjson_line_num + 1)

    records = []
    with open(ndjson_path_for_loading, "r", encoding="utf-8") as f0:
        # Skip lines before the start line
        for _ in range(start_line - 1):
            try:
                next(f0)
            except StopIteration:  # Handle case where start_line > total lines
                break
        # Read the lines assigned to this process
        for _ in range(start_line, end_line):
            try:
                line = next(f0)
                record: dict = parse_one_line(line, use_filter=use_filter)
                if record is not None:  # Check if parsing was successful
                    records.append(record)
            except StopIteration:  # Reached end of file prematurely
                break

    return records


def mpi_v3_subprocess(
        input_ndjson_path,
        ndjson_line_num,
        process_num,
        r,  # Assuming 'r' is the rank of the current process (0-based)
        use_filter=False,
):
    """
    Processes a chunk of an NDJSON file (assigned by rank) to aggregate scores
    by hour and by user ID. Calculates scores during the read process.

    Args:
        input_ndjson_path (str | Path): Path to the input NDJSON file.
        ndjson_line_num (int): Total number of lines in the file.
        process_num (int): Total number of MPI processes.
        r (int): Rank of the current process.
        use_filter (bool): Whether to apply filtering during line parsing.

    Returns:
        Tuple[dict, dict, list]:
            - hour_score (dict): Aggregated scores per hour.
              { 'YYYY-MM-DD HH:00': float_total_score, ... }
            - id_score (dict): Aggregated scores per user ID.
              { 'user_id_str': [float_total_score, str_username], ... }
            - failed_records (list): List of records that failed processing.
              [ record_dict_1, record_dict_2, ... ]
    """
    num_line_per_process = ceil(ndjson_line_num / process_num)
    # Calculate the line range [start, end) for this process (1-based indexing for lines)
    start_line = r * num_line_per_process + 1
    # The end line index is exclusive
    end_line = min(start_line + num_line_per_process, ndjson_line_num + 1)

    hour_score: dict = {}
    id_score: dict = {}  # <<< Initialize id_score dictionary
    failed_records = []

    with open(input_ndjson_path, "r", encoding="utf-8") as f0:
        # Skip lines before the start line
        for _ in range(start_line - 1):
            try:
                next(f0)
            except StopIteration:
                print(
                    f"Rank {r}: Warning"
                    f" - StopIteration encountered while skipping lines before start line {start_line}."
                )
                break  # Exit skip loop if file ends prematurely

        # Read and process lines assigned to this process
        current_line_num = start_line
        for _ in range(start_line, end_line):
            try:
                line = next(f0)
                record = parse_one_line(line, use_filter=use_filter)

                if record is None:  # Skip if parsing failed (e.g., empty line)
                    # print(f"Rank {r}: Warning - Skipped null record at line approx {current_line_num}")
                    current_line_num += 1
                    continue

                # --- Direct processing ---
                # Extract time, score, id, and username
                # Note: If any of these retrievals fail, the Exception block handles it.
                created_hour, sentiment_score = retrieve_time_and_score_from_a_record(
                    record=record,
                )
                id_0, username_0, _ = retrieve_id_name_score_from_a_record(  # We need id and username
                    record=record,
                )

                # --- Aggregate scores ---
                # Aggregate score by hour
                if created_hour not in hour_score:
                    hour_score[created_hour] = 0.0
                hour_score[created_hour] += sentiment_score

                # Aggregate score by user ID <<< Add id_score aggregation logic
                if id_0 not in id_score:
                    # Store score and username
                    id_score[id_0] = [sentiment_score, username_0]
                else:
                    # Add to existing score
                    id_score[id_0][0] += sentiment_score

            except StopIteration:  # Reached end of file within the processing loop
                print(
                    f"Rank {r}: Info - StopIteration encountered while processing line approx {current_line_num}."
                    f" Reached end of assigned chunk or file."
                )
                break  # Exit processing loop
            except Exception as e:
                # Log error and the problematic record
                print(f"Rank {r}: Error processing line {current_line_num}: {e}")
                # traceback.print_exc() # Optional: print full traceback
                # pprint.pprint(record) # Optional: print the failed record
                failed_records.append(record)

            current_line_num += 1  # Increment line counter regardless of success/failure within the try block

    # Return all three results <<< Update return statement
    return hour_score, id_score, failed_records


def mpi_v4_subprocess(file_path, use_filter=False):
    """
    Processes a single NDJSON file (presumably a piece from a larger dataset)
    to aggregate scores by hour and by user ID.

    Args:
        file_path (str | Path): Path to the NDJSON file piece.
        use_filter (bool): Whether to apply filtering during line parsing.

    Returns:
        Tuple[dict, dict, list]:
            - hour_score (dict): Aggregated scores per hour.
              { 'YYYY-MM-DD HH:00': float_total_score, ... }
            - id_score (dict): Aggregated scores per user ID.
              { 'user_id_str': [float_total_score, str_username], ... }
            - failed_records (list): List of records that failed processing.
              [ record_dict_1, record_dict_2, ... ]
    """
    hour_score = {}
    id_score = {}
    failed_records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            # Parse a single line
            record = parse_one_line(line, use_filter=use_filter)
            # If parse_one_line() returns None, likely an empty line or parsing error, skip it.
            if not record:
                continue

            # Try to extract required fields
            try:
                created_hour, sentiment_score = retrieve_time_and_score_from_a_record(
                    record=record,
                )
                # Assuming retrieve_id_name_score_from_a_record also exists and works similarly
                id_0, username_0, _ = retrieve_id_name_score_from_a_record(  # We only need id and username here
                    record=record,
                )
            except Exception as e:
                # If the record is missing key fields or another error occurs, add it to the failed list.
                print(f"[{file_path}] Error processing line {idx}: {e}")
                traceback.print_exc()
                pprint.pprint(record)
                failed_records.append(record)
                continue  # Skip to the next line
            else:
                # Aggregate score by hour
                if created_hour not in hour_score:
                    hour_score[created_hour] = sentiment_score
                else:
                    hour_score[created_hour] += sentiment_score

                # Aggregate score by user ID, storing the username as well
                if id_0 not in id_score:
                    # Store score and username (username only needs to be stored once)
                    id_score[id_0] = [sentiment_score, username_0]
                else:
                    # Add to existing score
                    id_score[id_0][0] += sentiment_score

    return hour_score, id_score, failed_records


def retrieve_time_and_score_from_a_record(record):
    """
    Extracts creation time (formatted to the hour) and sentiment score from a record.

    Args:
        record (dict): A dictionary representing a single record, expected structure:
            {
              "doc": {
                "createdAt": "ISO_timestamp_str",
                "sentiment": float_score,
                "account": { ... }
              }
            }

    Returns:
        tuple[str, float]: A tuple containing:
            - created_hour (str): Time formatted as 'YYYY-MM-DD HH:00'.
            - sentiment_score (float): The sentiment score.

    Raises:
        KeyError: If essential keys ('doc', 'createdAt', 'sentiment') are missing.
        TypeError: If 'doc' is not a dictionary.
        ValueError: If 'createdAt' is not a valid ISO timestamp or 'sentiment' cannot be cast to float.
    """
    if "doc" not in record:
        raise KeyError('Missing key "doc" in record')
    if not isinstance(record["doc"], dict):
        raise TypeError('record["doc"] is not of dict type')
    if "createdAt" not in record["doc"]:
        raise KeyError('Missing key "createdAt" in record["doc"]')
    if "sentiment" not in record["doc"]:
        raise KeyError('Missing key "sentiment" in record["doc"]')

    try:
        created_hour: str = high_level_api_to_convert_raw_time_to_preferred_str(
            record["doc"]["createdAt"]
        )
        # Ensure sentiment score is a float
        sentiment_score = float(record["doc"]["sentiment"])
    except ValueError as e:
        raise ValueError(f"Error converting time or sentiment: {e}") from e

    return created_hour, sentiment_score


def retrieve_id_name_score_from_a_record(record):
    """
    Extracts user ID, username, and sentiment score from a record.

    Args:
        record (dict): A dictionary representing a single record, expected structure:
            {
              "doc": {
                "createdAt": "...",
                "sentiment": float_score,
                "account": {
                  "id": "user_id_str",
                  "username": "username_str"
                }
              }
            }

    Returns:
        Tuple[str, str, float]: A tuple containing:
            - id_0 (str): The user ID.
            - username_0 (str): The username.
            - sentiment_0 (float): The sentiment score.

    Raises:
        KeyError: If essential keys ('doc', 'account', 'id', 'username', 'sentiment') are missing.
        TypeError: If 'doc' or 'account' are not dictionaries.
        ValueError: If 'sentiment' cannot be cast to float.
    """
    # Ensure doc exists and is a dict
    if "doc" not in record:
        raise KeyError('Missing key "doc" in record')
    if not isinstance(record.get("doc"), dict):  # Use .get for safer access before check
        raise TypeError('record["doc"] is not of dict type')

    doc = record["doc"]

    # Ensure account exists and is a dict
    if "account" not in doc:
        raise KeyError('Missing key "account" in record["doc"]')
    if not isinstance(doc.get("account"), dict):  # Use .get
        raise TypeError('record["doc"]["account"] is not of dict type')

    account = doc["account"]

    # Ensure required fields exist within account and doc
    if "id" not in account:
        raise KeyError('Missing key "id" in record["doc"]["account"]')
    if "username" not in account:
        raise KeyError('Missing key "username" in record["doc"]["account"]')
    if "sentiment" not in doc:
        raise KeyError('Missing key "sentiment" in record["doc"]')

    try:
        # Extract and potentially type-cast values
        id_0 = str(account["id"])  # Ensure ID is string
        username_0 = str(account["username"])  # Ensure username is string
        sentiment_0 = float(doc["sentiment"])  # Ensure sentiment is float
    except ValueError as e:
        raise ValueError(f"Error converting sentiment to float: {e}") from e
    except TypeError as e:
        # e.g. if id or username are unexpectedly not stringifiable
        raise TypeError(f"Error converting ID or username to string: {e}") from e

    return id_0, username_0, sentiment_0


def measure_time(func):
    """A decorator to measure the execution time of a function using MPI.Wtime()."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = MPI.Wtime()
        result = func(*args, **kwargs)
        end_time = MPI.Wtime()
        elapsed_time = end_time - start_time
        running_info = f"Function '{func.__name__}' executed in {elapsed_time:.5f} seconds"
        return result, running_info

    return wrapper


def split_file(
        file_path,
        total_line_num,
        to_pieces_num,
        output_folder,
        use_filter=False
):
    """Splits a large NDJSON file into smaller pieces.

    Args:
        file_path (str | Path): Path to the input NDJSON file.
        total_line_num (int): Total number of lines in the input file.
        to_pieces_num (int): The number of pieces to split the file into.
        output_folder (str | Path): Path to the folder where output pieces will be saved.
        use_filter (bool): Whether to apply filtering while reading lines.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not isinstance(output_folder, Path):
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    lines_per_file = ceil(total_line_num / to_pieces_num)
    print(
        f"Splitting {file_path} ({total_line_num} lines) into {to_pieces_num} pieces (~{lines_per_file} lines each)...")

    try:
        with open(file_path, "r", encoding="utf-8") as f0:
            current_line_global_idx = 0  # Keep track of global line number for error reporting
            for i in range(to_pieces_num):
                # Calculate exact lines for this piece (last piece might have fewer)
                lines_for_this_piece = min(lines_per_file, total_line_num - (i * lines_per_file))
                if lines_for_this_piece <= 0:
                    continue  # No more lines left

                output_filename = f"{file_path.stem}_piece_{i}{file_path.suffix}"
                output_path = output_folder / output_filename
                print(f"  Writing piece {i} ({lines_for_this_piece} lines) to {output_path}...")

                with open(output_path, "w", encoding="utf-8") as f1:
                    for _ in range(lines_for_this_piece):
                        try:
                            line = next(f0)
                            current_line_global_idx += 1
                            record = parse_one_line(line, use_filter=use_filter)
                            if record is not None:  # Only write if parsing succeeds
                                write_line = dict_to_a_line(record)
                                f1.write(write_line)
                        except StopIteration:
                            print(f"Warning: Reached end of file unexpectedly while writing piece {i}.")
                            break  # Stop writing for this piece
                        except Exception as e:
                            print(f"Error processing line {current_line_global_idx} in {file_path}: {e}")
                            traceback.print_exc()
                            # Optionally write the problematic line/record to an error file
                            continue  # Skip this line and continue with the piece
                print(f"  Finished piece {i}.")
        print("File splitting completed.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during file splitting: {e}")
        traceback.print_exc()


def check_split_files_exist(
        original_file_path,
        to_pieces_num,
        output_folder
) -> bool:
    """Checks if all expected split files generated by split_file exist.

    Args:
        original_file_path (str | Path): Path to the *original* input file that was split.
        to_pieces_num (int): The number of pieces the file was supposed to be split into.
        output_folder (str | Path): Path to the folder where output pieces should be located.

    Returns:
        bool: True if all expected split files exist, False otherwise.
    """
    if not isinstance(original_file_path, Path):
        original_file_path = Path(original_file_path)
    if not isinstance(output_folder, Path):
        output_folder = Path(output_folder)

    # First, check if the output folder exists. If not, files can't exist.
    if not output_folder.is_dir():
        # print(f"Info: Output folder '{output_folder}' does not exist.")
        return False

    if to_pieces_num <= 0:
        # print("Info: Number of pieces is non-positive. Checking for 0 files (always true).")
        return True  # If 0 pieces were expected, then the condition is met.

    all_exist = True
    for i in range(to_pieces_num):
        # Construct the expected filename based on the logic in split_file
        expected_filename = f"{original_file_path.stem}_piece_{i}{original_file_path.suffix}"
        expected_file_path = output_folder / expected_filename

        # Check if this specific file exists and is actually a file
        if not expected_file_path.is_file():
            # print(f"Info: Missing expected split file: {expected_file_path}") # Optional: uncomment for debugging
            all_exist = False
            break  # No need to check further if one is missing

    return all_exist


def mpi_v4_single_process_version():
    pass


def tst():
    """Placeholder for testing purposes."""
    pass
