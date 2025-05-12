import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import pickle
from PIL import Image as PIL_Image
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import time


# Configure GPU memory growth
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def path_join(dirname, filenames):
    """Joins a directory path with a list of filenames."""
    return [os.path.join(dirname, filename) for filename in filenames]

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

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    """Plots up to 9 images with their true and predicted labels."""
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

def example_errors(cls_true, cls_pred):
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
                cls_pred=cls_pred[incorrect_indices]
            )

    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    plot_confusion_matrix(cm=cm, classes=class_names)

def plot_training_history(history, history_fine=None):
    """Plots training and validation accuracy and loss, including fine-tuning if provided."""
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

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(total_epochs), acc, label='Training Accuracy')
    plt.plot(range(total_epochs), val_acc, label='Validation Accuracy')
    if history_fine:
        plt.axvline(x=epochs_initial-0.5, linestyle='--', label='Start Fine-Tuning')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(total_epochs), loss, label='Training Loss')
    plt.plot(range(total_epochs), val_loss, label='Validation Loss')
    if history_fine:
        plt.axvline(x=epochs_initial-0.5, linestyle='--', label='Start Fine-Tuning')
    plt.legend()
    plt.title('Loss')
    plt.show()

def open_video(video):
    """Opens a video file for processing."""
    return cv2.VideoCapture(video)

def check_video(cap, video_path):
    """Verifies if a video can be opened and read."""
    if not os.path.exists(video_path):
        print(f"Error: Video not found at '{video_path}'")
        return False
    if not cap.isOpened():
        print(f"Error: Cannot open video at '{video_path}'")
        return False
    success, _ = cap.read()
    if not success:
        print(f"Error: Cannot read first frame of '{video_path}'")
        cap.release()
        return False
    return True

def close_cap(cap):
    """Releases a video capture object."""
    cap.release()

def getFrameRate(video):
    """Retrieves the frame rate of a video."""
    return video.get(cv2.CAP_PROP_FPS)

def generate_out_videoname(vid_base):
    """Generates a standardized video name from a base name."""
    if "video-" in vid_base:
        return vid_base.split('.')[0]
    try:
        collision_num = vid_base.split('collision')[1]
        return f"video-{collision_num.zfill(5)}"
    except IndexError:
        print(f"Warning: Invalid video name format '{vid_base}'. Using default.")
        return 'video-00001'

def generate_framename(video_num, pos_frame):
    """Generates a frame name from video number and frame position."""
    return f"video-{str(video_num).zfill(5)}-frame-{str(pos_frame).zfill(5)}"

def generate_video_num(out_videoname):
    """Extracts the video number from a video name."""
    return int(out_videoname.split('-')[1])

def generate_paired_file_lists(video_prefix='collision', excel_prefix='video-', range_min=1, range_max=94):
    """Generates paired lists of video and Excel file paths."""
    video_files = []
    excel_files = []
    video_dir = os.path.join(os.getcwd(), 'videos')
    excel_dir = os.path.join(os.getcwd(), 'dataframes')
    
    for i in range(range_min, range_max + 1):
        video_files.append(os.path.join(video_dir, f'{video_prefix}{str(i).zfill(2)}.mp4'))
        excel_files.append(os.path.join(excel_dir, f'{excel_prefix}{str(i).zfill(5)}.xlsx'))
    
    return video_files, excel_files

def create_image_directories(base_dir='image_data'):
    """Creates train and test directories for image data."""
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    for dir_path in [train_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, '0'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, '1'), exist_ok=True)
    return train_dir, test_dir

def process_and_save_frames(excel_files, video_files, output_dir, target_size=(224, 224), train_ratio=0.8):
    """Processes video frames, saves them to directories, and returns frame data."""
    frames, labels, filenames = [], [], []
    
    for excel_file, video_file in zip(excel_files, video_files):
        if not (os.path.exists(excel_file) and os.path.exists(video_file)):
            print(f"Skipping: File missing - Excel: {excel_file}, Video: {video_file}")
            continue

        df = pd.read_excel(excel_file)
        cap = open_video(video_file)
        if not check_video(cap, video_file):
            continue

        frame_dict = {row['file']: int(row['collision']) for _, row in df.iterrows()}
        video_base = os.path.basename(video_file).split('.')[0]
        out_video_name = generate_out_videoname(video_base)
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_name = f"{out_video_name}-frame-{str(frame_count + 1).zfill(5)}"
            if frame_name in frame_dict:
                label = frame_dict[frame_name]
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                
                split = 'train' if np.random.rand() < train_ratio else 'test'
                save_path = os.path.join(output_dir, split, str(label), f"{frame_name}.png")
                # Convert to uint8 before color conversion and saving
                frame_uint8 = (frame * 255).astype(np.uint8)
                cv2.imwrite(save_path, cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
                
                frames.append(frame)
                labels.append(label)
                filenames.append(save_path)
            
            frame_count += 1
        close_cap(cap)
    
    return np.array(frames), np.array(labels), filenames

# --- Main Script ---
# Setup directories
output_base_dir = 'image_data'
train_dir, test_dir = create_image_directories(output_base_dir)

# Load and process data
video_files, excel_files = generate_paired_file_lists(range_min=70, range_max=93)
all_frames, all_labels, all_filenames = process_and_save_frames(
    excel_files, video_files, output_base_dir, target_size=(224, 224)
)

# --- VGG16 Model Setup ---
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# --- Data Generators ---
batch_size = 20
datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.9, 1.1],
    fill_mode='nearest'
)
datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

generator_train = datagen_train.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse', shuffle=True
)
generator_test = datagen_test.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse', shuffle=False
)

steps_test = int(np.ceil(generator_test.n / batch_size))
class_names = list(generator_train.class_indices.keys())

# --- Class Weights ---
class_weight = dict(enumerate(
    compute_class_weight(class_weight='balanced', classes=np.unique(generator_train.classes), y=generator_train.classes)
))

# --- Initial Training ---
base_model.trainable = False
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
epochs = 20
steps_per_epoch = int(np.ceil(generator_train.n / batch_size))

history = model.fit(
    generator_train,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=generator_test,
    validation_steps=steps_test,
    class_weight=class_weight
)

# --- Fine-Tuning ---
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = 'block5' in layer.name or 'block4' in layer.name

model.compile(optimizer=Adam(learning_rate=2e-8), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(
    generator_train,
    epochs=epochs * 2,
    initial_epoch=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=generator_test,
    validation_steps=steps_test,
    class_weight=class_weight
)

# --- Evaluation ---
model.evaluate(generator_test, steps=steps_test)
generator_test.reset()
y_pred = model.predict(generator_test, steps=steps_test)
cls_pred = np.argmax(y_pred, axis=1)
cls_true = generator_test.classes
example_errors(cls_true, cls_pred)

# --- Save Results ---
timestamp = int(time.time())
name = f"VGG-collision-avoidance-{timestamp}"
plot_training_history(history, history_fine)
plt.gcf().savefig(f"{name}.pdf", bbox_inches='tight')

model.save(f"models/{name}.keras")
with open('trainHistoryDict_fine.pkl', 'wb') as f:
    pickle.dump(history_fine.history, f)
model.save_weights(f"models/{name}_weights.weights.h5")

session.close()