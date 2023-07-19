import os
import pandas as pd
import numpy as np

from IPython.display import display

class Data:
    """
    Class that handles operations on data (ex: loading, preprocessing, etc.)
    """
    
    @staticmethod
    def dataframe_to_np_array(df: pd.DataFrame) -> np.array:
        """
        Convert a dataframe to a numpy array
        Params:
            df (pd.DataFrame): Dataframe to convert to numpy array.
        Returns:
            np.array: Numpy array containing the data from the dataframe, with the correct shape.
        """
        # TODO: Implement this method
        pass

    @staticmethod
    def load_class_data(base_dir: str, class_name: str) -> pd.DataFrame:
        """
        Load data from a given class directory. Here the name of the class is the name of the directory.
        Params:
            base_dir (str): Base directory of the data
            class_name (str): Name of the class
        Returns:
            pd.DataFrame: Dataframe containing the data from the class
        """
        class_dir: str = os.path.join(base_dir, class_name)
        super_df: pd.Dataframe = None
        
        for file_name in os.listdir(class_dir):
            file_path: str = os.path.join(class_dir, file_name)
            new_data: pd.Dataframe = pd.read_csv(file_path)
            
            super_df = pd.concat([super_df, new_data], ignore_index=False) if super_df is not None else new_data

        return super_df

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
            pass
    
        return data