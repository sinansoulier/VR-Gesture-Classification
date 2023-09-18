import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Plot:
    """
    Class that contains methods for plotting data.
    """
    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray, fig_size: tuple[int, int]) -> None:
        """
        Plot the confusion matrix.
        Params:
            confusion_matrix (np.ndarray): Confusion matrix.
            fig_size (tuple[int, int]): Figure size.
        """
        plt.figure(figsize=fig_size)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()