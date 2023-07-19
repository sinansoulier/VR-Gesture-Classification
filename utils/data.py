import os
import pandas as pd
import numpy as np

from IPython.display import display

class Data:
    """
    Class that handles operations on data (ex: loading, preprocessing, etc.)
    """

    @staticmethod
    def truncate_dataframe_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
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
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
    def load_class_data(base_dir: str, class_name: str) -> list[pd.DataFrame]:
        """
        Load data from a given class directory. Here the name of the class is the name of the directory.
        Params:
            base_dir (str): Base directory of the data
            class_name (str): Name of the class
        Returns:
            pd.DataFrame: Dataframe containing the data from the class
        """
        class_dir: str = os.path.join(base_dir, class_name)

        dataframe_list: list[pd.DataFrame] = []
        
        for file_name in os.listdir(class_dir):
            # Define the full path of the file
            file_path: str = os.path.join(class_dir, file_name)
            # Load the data from the file into a dataframe
            new_data: pd.Dataframe = pd.read_csv(file_path)
            # Process the dataframe and add it to the list
            dataframe_list.append(Data.process_dataframe(new_data))

        return dataframe_list

    @staticmethod
    def load_data(base_dir: str) -> list[pd.DataFrame]:
        """
        Load data from a given base directory. Here the name of the class is the name of the directory.
        Params:
            base_dir (str): Base directory of the data
        Returns:
            list[pd.DataFrame]: List of dataframes containing the data from all the classes.
        """
        data: list[pd.DataFrame] = []
        
        for class_name in os.listdir(base_dir):
            data.append(Data.load_class_data(base_dir, class_name))
            # FIXME: implement loop body
            pass
    
        return data