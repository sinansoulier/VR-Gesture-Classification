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
        df = df.applymap(lambda x: [float(elt) for elt in x.split(';')])

        column_names = df.columns
        list_df_ax = []

        # Apply processing to earch axis
        for i, axis in enumerate(['x', 'y', 'z']):
            new_column_names = column_names.map(lambda x: x + '_' + axis)
            df_ax = df.applymap(lambda x: x[i])
            df_ax.columns = new_column_names
            list_df_ax.append(df_ax)

        # Concatenate the newly created dataframes
        df = pd.concat(list_df_ax, axis=1)
    
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
            # Add the class name to the build the classes list
            classes.append(class_name)
            # Load the data for the current class
            class_data = Data.__load_class_data(base_dir, class_name)
            # Add the data to the list of dataframes
            data.extend(class_data)
            # Add the class name to the labels list
            labels.extend([class_name] * len(class_data))
    
        # Return the dataframes as a numpy array, the labels as a numpy array and the classes as a numpy array
        return Data.__convert_to_numpy(data), np.array(labels), np.array(classes)