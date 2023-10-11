import os
from typing import Generator

import pandas as pd
import numpy as np
import torch

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
    def __extend_dataframe(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        """
        Extend a dataframe to a given number of rows.
        Params:
            df (pd.DataFrame): Dataframe to extend
            max_rows (int): Maximum number of rows to reach
        Returns:
            pd.DataFrame: Extended dataframe
        """
        nb_rows_to_add = max_rows - df.shape[0]

        if nb_rows_to_add > 0:
            df = pd.concat([df, pd.DataFrame(np.zeros((nb_rows_to_add, df.shape[1])), columns=df.columns)], ignore_index=True, axis=0)
        
        return df
    
    @staticmethod
    def __create_new_columns(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Create a new column from a given column by splitting it into 3 columns (x, y, z).
        Params:
            df (pd.DataFrame): Dataframe to process
            column_name (str): Name of the column to split
        Returns:
            pd.DataFrame: Dataframe with the new columns
        """
        return pd.DataFrame(
            df[column_name].to_list(),
            columns=[column_name + '_x', column_name + '_y', column_name + '_z']
        )
        
    @staticmethod
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataframe by splitting each column into 3 columns (x, y, z) for each column.
        Params:
            df (pd.DataFrame): Dataframe to process
        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Drop the time column
        df.drop(columns=['Time'], inplace=True)
        # Cast all data to float
        df = df.map(lambda x: [float(elt) for elt in x.split(';')])

        # Apply split on each column
        list_df_ax = df.columns.map(lambda column_name: Data.__create_new_columns(df, column_name))

        # Concatenate the newly created dataframes
        df = pd.concat(list_df_ax, axis=1)
    
        return df

    @staticmethod
    def truncate_to_numpy(list_df: list[pd.DataFrame]) -> np.ndarray:
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
        ).astype(np.float32)
        
        return truncated_list_df
    
    @staticmethod
    def convert_to_numpy(list_df: list[pd.DataFrame]) -> np.ndarray:
        """
        Truncate dataframes if needed, then convert them to numpy arrays.
        Params:
            list_df (list[pd.DataFrame]): List of dataframes to truncate.
        Returns:
            list[np.ndarray]: List of truncated dataframes under the form of a numpy multidimensional array.
        """
        # Get the maximum number of rows for all dataframes
        max_rows: int = max([df.shape[0] for df in list_df])

        # Extend the dataframes if needed
        np_extended_data: np.ndarray = np.array(
            list(map(lambda df: Data.__extend_dataframe(df, max_rows).to_numpy().astype(np.float32), list_df)),
            dtype=object
        ).astype(np.float32)
        
        return np_extended_data
    
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
            dataframe_list.append(Data.process_dataframe(new_data))

        return dataframe_list

    @staticmethod
    def load_data(base_dir: str) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray]:
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
        
        list_paths = os.listdir(base_dir)
        # list_paths = np.array(list_paths)[['HORIZONTAL' in elt for elt in list_paths]]

        for i, class_name in enumerate(list_paths):
            print(i, class_name)
            # Add the class name to the build the classes list
            classes.append(class_name)
            # Load the data for the current class
            class_data = Data.__load_class_data(base_dir, class_name)
            # Add the data to the list of dataframes
            data.extend(class_data)
            # Add the class name to the labels list
            labels.extend([i] * len(class_data))

        # Return the dataframes as a numpy array, 
        # the labels as a numpy array and the classes as a numpy array
        return data, np.array(labels), np.array(classes)
    
    @staticmethod
    def labeled_data_generator(X: np.ndarray, y: np.ndarray, batch_size: int=32) -> Generator:
        """
        Data generator for PyTorch.
        Params:
            X (np.ndarray): Data
            y (np.ndarray): Labels
            batch_size (int): Size of the batch, default to 32
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the data and the labels
        """
        # Generate indices
        indices: np.ndarray = np.arange(X.shape[0])
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Retrieve suffled data
        X, y = X[indices], y[indices]

        # Iterate over the dataset
        for i in range(0, X.shape[0], batch_size):
            # Get batch data
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            # Convert to torch tensors
            X_batch = torch.from_numpy(X_batch)
            y_batch = torch.from_numpy(y_batch)
            
            # Yield the batch
            yield X_batch, y_batch
    
    @staticmethod
    def unlabeled_data_generator(X: np.ndarray, batch_size: int=32) -> Generator:
        """
        Data generator for PyTorch.
        Params:
            X (np.ndarray): Data
            batch_size (int): Size of the batch, default to 32
        Returns:
            torch.Tensor: Data
        """
        # Generate indices
        indices: np.ndarray = np.arange(X.shape[0])
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Retrieve suffled data
        X = X[indices]

        # Iterate over the dataset
        for i in range(0, X.shape[0], batch_size):
            # Get batch data
            X_batch = X[i:i+batch_size]
            
            # Convert to torch tensors
            X_batch = torch.from_numpy(X_batch)
            
            # Yield the batch
            yield X_batch