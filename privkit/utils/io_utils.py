"""IO utility methods."""
import pandas as pd

from pathlib import Path
from privkit.utils import dev_utils as du


def read_dataframe(filepath_or_buffer: str or Path or object, unique: bool = True, extension: str = None, **kwargs):
    """
    Reads data to a pandas dataframe object.

    :param str or Path or object filepath_or_buffer: either a path to a file (a str or Path) or any object that can be read from pandas read() methods.
    :param bool unique: if True, this method is the unique method to read dataframe. If False, data can be read with
        other method. The default value is `True`.
    :param str extension: specify the file extension. Default value is None.
    :param kwargs: the keyword arguments are used for specifying arguments according to the Pandas read methods.
    :return: pandas dataframe object or the parameter `filepath_or_buffer`.
    """
    if not extension and (isinstance(filepath_or_buffer, str) or isinstance(filepath_or_buffer, Path)):
        extension = ''.join(Path(filepath_or_buffer).suffixes)

    if extension:
        if extension == '.csv' or extension == 'csv':
            return pd.read_csv(filepath_or_buffer, **(kwargs or {}))
        elif extension == '.pkl' or extension == 'pkl':
            return pd.read_pickle(filepath_or_buffer, **(kwargs or {}))
        elif extension == '.json' or extension == 'json':
            return pd.read_json(filepath_or_buffer, **(kwargs or {}))
        else:
            return pd.DataFrame(filepath_or_buffer, **(kwargs or {}))
    else:
        for code in (
                lambda: pd.read_pickle(filepath_or_buffer, **(kwargs or {})),
                lambda: pd.read_csv(filepath_or_buffer, **(kwargs or {})),
                lambda: pd.read_json(filepath_or_buffer, **(kwargs or {})),
                lambda: pd.read_table(filepath_or_buffer, **(kwargs or {})),
                lambda: pd.DataFrame(filepath_or_buffer, **(kwargs or {}))
        ):
            try:
                return code()
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    raise FileNotFoundError(f"FileNotFoundError: {e}")
                pass

    if unique:
        raise TypeError("Data format is not supported to convert to a Pandas dataframe.")
    else:
        return filepath_or_buffer


def write_dataframe(data2save: pd.DataFrame, filepath: str, filename: str, extension: str = 'pkl'):
    """
    Writes a dataframe to a file.

    :param pd.DataFrame data2save: dataframe to save.
    :param str filepath: path where data should be saved.
    :param str filename: name of the file to be saved.
    :param str extension: extension of the format of how the file should be saved. The default value is `'pkl'`.
    """
    if extension == '.json' or extension == 'json':
        filepath = "{}{}.json".format(filepath, filename, extension)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data2save.to_json(filepath)
    elif extension == '.csv' or extension == 'csv' or extension == '.txt' or extension == 'txt':
        filepath = "{}{}.csv".format(filepath, filename, extension)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data2save.to_csv(filepath)
    else:
        filepath = "{}{}.pkl".format(filepath, filename)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data2save.to_pickle(filepath)
    du.log('File saved at {}.'.format(filepath))
