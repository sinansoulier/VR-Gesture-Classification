import os
import pandas as pd
import numpy as np

class Data:
    """
    Class that handles operations on data (ex: loading, preprocessing, etc.)
    """

    @staticmethod
    def __truncate_dataframe_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        """
        Truncate a dataframe to a given number of rows.
        Params:
            df (pd.DataFrame): Dataframe to truncate
            max_rows (int): Maximum number of rows to keep
        Returns:
            pd.DataFrame: Truncated dataframe
        """
        return df.iloc[:max_rows]
    
    @staticmethod
    def __process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        # Drop the time column
        df: pd.DataFrame = df.drop(columns=['Time'])

        # For each column, split the string into 3 columns (x, y, z)
        for column_name in df.columns:
            # Convert the string column to floats
            column_to_float: pd.Series = df[column_name].apply(lambda x: np.array(x.split(';')).astype(np.float32))
            
            # Create 3 new columns (x, y, z)
            column_x: pd.Series = column_to_float.map(lambda x: x[0])
            column_x.name = column_name + '_x'

            column_y: pd.Series = column_to_float.map(lambda x: x[1])
            column_y.name = column_name + '_y'

            column_z: pd.Series = column_to_float.map(lambda x: x[2])
            column_z.name = column_name + '_z'

            # Add newly created columns to the dataframe
            df = pd.concat([df, column_x, column_y, column_z], axis=1)

            # Drop the origin column
            df = df.drop(columns=[column_name])
    
        return df

    @staticmethod
    def __convert_to_numpy(list_df: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Truncate dataframes if needed, then convert them to numpy arrays.
        Params:
            list_df (list[pd.DataFrame]): List of dataframes to truncate.
        Returns:
            list[np.ndarray]: List of truncated dataframes under the form of a numpy multidimensional array.
        """
        # Get the minimum number of rows for all dataframes
        min_rows: int = min([df.shape[0] for df in list_df])
        
        # Truncate the dataframes if needed
        truncated_list_df: np.ndarray = np.array(
            list(map(lambda df: Data.__truncate_dataframe_rows(df, min_rows).to_numpy().astype(np.float32), list_df)),
            dtype=object
        )
        
        return truncated_list_df
    
    @staticmethod
    def __load_class_data(base_dir: str, class_name: str) -> list[pd.DataFrame]:
        """
        Load data from a given class directory. Here the name of the class is the name of the directory.
        Params:
            base_dir (str): Base directory of the data
            class_name (str): Name of the class
        Returns:
            list[pd.DataFrame]: List of dataframes containing all the data from the given class.
        """
        # Determine given class data directory
        class_dir: str = os.path.join(base_dir, class_name)
        # List of dataframes containing the data from the class
        dataframe_list: list[pd.DataFrame] = []
        
        for file_name in os.listdir(class_dir):
            # Define the full path of the file
            file_path: str = os.path.join(class_dir, file_name)
            # Load the data from the file into a dataframe
            new_data: pd.Dataframe = pd.read_csv(file_path)
            # Process the dataframe and add it to the list
            dataframe_list.append(Data.__process_dataframe(new_data))

        return dataframe_list

    @staticmethod
    def load_data(base_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from a given base directory. Here the name of the class is the name of the directory.
        Params:
            base_dir (str): Base directory of the data
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the dataframes as a numpy array,
                                                       the labels as a numpy array,
                                                       and the classes as a numpy array
        """
        data: list[pd.DataFrame] = []
        classes: list[str] = []
        labels: list[str] = []
        
        for class_name in os.listdir(base_dir):
            classes.append(class_name)
            class_data = Data.__load_class_data(base_dir, class_name)
            data.extend(class_data)
            labels.extend([class_name] * len(class_data))
    
        # Return the dataframes as a numpy array, the labels as a numpy array and the classes as a numpy array
        return Data.__convert_to_numpy(data), np.array(labels), np.array(classes)