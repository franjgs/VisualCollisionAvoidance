#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:48:47 2025

@author: fran
"""

# plotting_utils.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import cv2

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues,
                          bal_acc=None, pfa=None, pd=None, save_path=None):
    """Plots a confusion matrix, with optional normalization, and includes metrics in title.

    Args:
        cm (np.ndarray): The confusion matrix.
        classes (list): List of class names (e.g., ['no_collision', 'collision']).
        normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
        title (str, optional): Base title for the plot. Defaults to 'Confusion Matrix'.
        cmap (matplotlib.colors.Colormap, optional): Colormap for the plot. Defaults to plt.cm.Blues.
        bal_acc (float, optional): Balanced Accuracy to display in the title.
        pfa (float, optional): Probability of False Alarm to display in the title.
        pd (float, optional): Probability of Detection (True Positive Rate) to display in the title.
        save_path (str, optional): Path to save the plot as a PDF. Defaults to None.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6)) # Increased figure size for better readability of metrics
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    full_title = title
    if bal_acc is not None and pfa is not None and pd is not None:
        full_title += f'\nBalAcc: {bal_acc:.2f}, pFA: {pfa:.2f}, pD: {pd:.2f}'

    plt.title(full_title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        # Ensure the directory for saving exists
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Saved confusion matrix plot to {save_path}")
    plt.show() # Keep plt.show() to display the plot during execution


def plot_images(images, cls_true, class_names, cls_pred=None, smooth=True, save_path=None, model_name=None):
    """Plots up to 9 images with their true and predicted labels.

    Args:
        images (np.ndarray): Array of images to plot.
        cls_true (np.ndarray): Array of true class labels (indices).
        class_names (list): List of class names (strings).
        cls_pred (np.ndarray, optional): Array of predicted class labels (indices). Defaults to None.
        smooth (bool, optional): Whether to use smooth interpolation. Defaults to True.
        save_path (str, optional): Base path to save the plot. Defaults to None.
        model_name (str, optional): Name of the model for the filename. Defaults to None.
    """
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.6 if cls_pred is not None else 0.3, wspace=0.3)
    interpolation = 'spline16' if smooth else 'nearest'

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], interpolation=interpolation)
            cls_true_name = class_names[cls_true[i]]
            xlabel = f"True: {cls_true_name}"
            if cls_pred is not None:
                cls_pred_name = class_names[cls_pred[i]]
                xlabel += f"\nPred: {cls_pred_name}"
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

    if save_path:

        # Ensure the directory for saving exists
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Saved image plot to {save_path}")
    plt.show()

def load_images(image_paths):
    """Loads images from a list of file paths. Converts from BGR (OpenCV) to RGB (Matplotlib)."""
    loaded_images = []
    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                loaded_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                print(f"Warning: Could not read image file: {path}")
        else:
            print(f"Warning: Image file not found: {path}")
    return np.array(loaded_images)


def example_errors(cls_true, cls_pred, generator_test, class_names, output_dir=None, model_name=None):
    """Visualizes misclassified images and plots the confusion matrix, including metrics.

    Args:
        cls_true (np.ndarray): Array of true class labels (indices).
        cls_pred (np.ndarray): Array of predicted class labels (indices).
        generator_test: Keras ImageDataGenerator (or similar) with .filepaths, used for getting image paths.
                        For multi-frame, this might not be directly applicable for image visualization.
        class_names (list): List of class names (strings).
        output_dir (str, optional): Directory to save plots.
        model_name (str, optional): Name of the model for plot titles/filenames.
    """
    print("Analyzing misclassifications...")
    incorrect = cls_true != cls_pred
    num_incorrect = np.sum(incorrect)
    print(f"Misclassified images: {num_incorrect}")

    if num_incorrect > 0:
        incorrect_indices = np.where(incorrect)[0][:9]
        # This part is primarily for single-frame models using ImageDataGenerator.
        # For multi-frame, 'generator_test.filepaths' might not exist or be relevant
        # if you're loading .npy sequences. You would need a different mechanism
        # to retrieve/display misclassified sequence frames.
        try:
            if hasattr(generator_test, 'filepaths'):
                misclassified_images = load_images(np.array(generator_test.filepaths)[incorrect_indices])
                if len(misclassified_images) > 0:
                    # Prepare save path for misclassified images plot
                    img_save_path = None
                    if output_dir:
                        if model_name:
                            img_filename = f'{model_name}_Misclassified_Images.pdf'
                        else:
                            img_filename = 'Misclassified_Images.pdf'
                        # model_name will be added by plot_images
                        img_save_path = os.path.join(output_dir, img_filename)

                    plot_images(
                        images=misclassified_images,
                        cls_true=cls_true[incorrect_indices],
                        class_names=class_names,
                        cls_pred=cls_pred[incorrect_indices],
                        save_path=img_save_path, # Pass the base save path
                        model_name=model_name # Pass the model name
                    )
            else:
                print("Note: Skipping visualization of misclassified images as 'generator_test.filepaths' is not available.")
        except Exception as e:
            print(f"Error visualizing misclassified images: {e}")


    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # Calculate metrics
    # Assuming binary classification where class_names[0] is negative ('no_collision'), class_names[1] is positive ('collision')
    # If your classes are ordered differently, adjust indices (e.g., cm[0,0] for positive, cm[1,1] for negative)
    TP = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0 # True Positives (Actual 1, Predicted 1)
    TN = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0 # True Negatives (Actual 0, Predicted 0)
    FP = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0 # False Positives (Actual 0, Predicted 1) - Type I error / False Alarm
    FN = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0 # False Negatives (Actual 1, Predicted 0) - Type II error / Missed Detection

    # Handle division by zero for robustness (e.g., if a class has no true samples)
    pD = TP / (TP + FN) if (TP + FN) > 0 else 0.0 # Probability of Detection (True Positive Rate/Recall)
    pFA = FP / (FP + TN) if (FP + TN) > 0 else 0.0 # Probability of False Alarm (False Positive Rate)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0 # True Negative Rate (Specificity)

    bal_acc = (pD + TNR) / 2.0 # Balanced Accuracy

    # Prepare save path for confusion matrix
    cm_save_path = None
    if output_dir:
        cm_filename = 'Confusion_Matrix.pdf'
        if model_name:
            cm_filename = f'{model_name}_Confusion_Matrix.pdf'
        cm_save_path = os.path.join(output_dir, cm_filename)

    plot_confusion_matrix(cm=cm, classes=class_names,
                          title=f'Confusion Matrix for {model_name}',
                          bal_acc=bal_acc, pfa=pFA, pd=pD,
                          save_path=cm_save_path)

def plot_training_history(history, model_name, history_fine=None, save_path=None):
    """Plots training and validation accuracy and loss, including fine-tuning if provided,
       and includes the model name in the plot title and/or filename.

    Args:
        history: The History object returned by model.fit().
        model_name (str): The name of the model.
        history_fine: Optional History object for fine-tuning.
        save_path (str): Path to save the plot.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_initial = len(acc)

    if history_fine:
        acc_fine = history_fine.history['accuracy']
        val_acc_fine = history_fine.history['val_accuracy']
        loss_fine = history_fine.history['loss']
        val_loss_fine = history_fine.history['val_loss']
        total_epochs = epochs_initial + len(acc_fine)
        acc = acc + acc_fine
        val_acc = val_acc + val_acc_fine
        loss = loss + loss_fine
        val_loss = val_loss + val_loss_fine
    else:
        total_epochs = epochs_initial

    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(total_epochs), acc, label='Training Accuracy')
    plt.plot(range(total_epochs), val_acc, label='Validation Accuracy')
    if history_fine:
        plt.axvline(x=epochs_initial - 0.5, linestyle='--', label='Start Fine-Tuning')
    plt.legend()
    if model_name:
        plt.title(f'{model_name} Accuracy')  # Include model name in title
    else:
        plt.title('Accuracy')
        
    plt.subplot(1, 2, 2)
    plt.plot(range(total_epochs), loss, label='Training Loss')
    plt.plot(range(total_epochs), val_loss, label='Validation Loss')
    if history_fine:
        plt.axvline(x=epochs_initial - 0.5, linestyle='--', label='Start Fine-Tuning')
    plt.legend()
    if model_name:
        plt.title(f'{model_name} Loss')  # Include model name in title
    else:
        plt.title('Loss')
        
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Saved plot to {save_path}")

    return fig

# --- Plot Training History ---
def plot_single_history(history, save_path=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()