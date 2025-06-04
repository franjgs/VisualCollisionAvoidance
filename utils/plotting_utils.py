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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plots a confusion matrix, with optional normalization."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
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
    plt.show()

def plot_images(images, cls_true, class_names, cls_pred=None, smooth=True):
    """Plots up to 9 images with their true and predicted labels.

    Args:
        images (np.ndarray): Array of images to plot.
        cls_true (np.ndarray): Array of true class labels (indices).
        class_names (list): List of class names (strings).
        cls_pred (np.ndarray, optional): Array of predicted class labels (indices). Defaults to None.
        smooth (bool, optional): Whether to use smooth interpolation. Defaults to True.
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

    plt.show()

def load_images(image_paths):
    """Loads images from a list of file paths."""
    return np.array([cv2.imread(path) for path in image_paths if os.path.exists(path)])

def example_errors(cls_true, cls_pred, generator_test, class_names):
    """Visualizes misclassified images and plots the confusion matrix."""
    print("Analyzing misclassifications...")
    incorrect = cls_true != cls_pred
    num_incorrect = np.sum(incorrect)
    print(f"Misclassified images: {num_incorrect}")

    if num_incorrect > 0:
        incorrect_indices = np.where(incorrect)[0][:9]
        misclassified_images = load_images(np.array(generator_test.filepaths)[incorrect_indices])
        if len(misclassified_images) > 0:
            plot_images(
                images=misclassified_images,
                cls_true=cls_true[incorrect_indices],
                class_names=class_names,  # Pass class_names here
                cls_pred=cls_pred[incorrect_indices]
            )

    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    plot_confusion_matrix(cm=cm, classes=class_names)

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
        # Include model name in filename
        base_path, ext = os.path.splitext(save_path)
        save_path_with_model_name = f"{base_path}_{model_name}{ext}"
        fig.savefig(save_path_with_model_name, bbox_inches='tight', format='pdf')
        print(f"Saved plot to {save_path_with_model_name}")

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