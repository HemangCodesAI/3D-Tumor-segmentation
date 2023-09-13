import streamlit as st
!pip install nibabel
import nibabel as nib
import tempfile
import numpy as np
import cv2
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from PIL import Image
# Load the training history CSV file
history_file_path = 'training_per_class.log'  # Update with your actual path
history_data = pd.read_csv(history_file_path)

# Load the model architecture images
model_architecture_images = {
    'Model Architecture': 'model_architecture.png',  # Update with your actual path
}
st.set_option('deprecation.showPyplotGlobalUse', False)
VOLUME_SLICES = 100
IMG_SIZE = 128
Volume_starts_at = 22
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    3 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    1 : 'ENHANCING' # original 4 -> converted into 3 later
}

def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
   #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
#    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss
 
# https://github.com/keras-team/keras/issues/9395
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)

# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def process_data(flair,t1ce):
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+Volume_starts_at], (IMG_SIZE,IMG_SIZE))
        X[j,:,:,1] = cv2.resize(t1ce[:,:,j+Volume_starts_at], (IMG_SIZE,IMG_SIZE))
    return X

def predict(data):
    model=keras.models.load_model('model_per_class.h5',custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                   "sensitivity":sensitivity,
                                                   "specificity":specificity,
                                                   "dice_coef_necrotic": dice_coef_necrotic,
                                                   "dice_coef_edema": dice_coef_edema,
                                                   "dice_coef_enhancing": dice_coef_enhancing
                                                  }, compile=False)
    return model.predict(data/np.max(data), verbose=1)

def show_output_in_streamlit1(flair, p, start_slice=60):
    original = flair
    core = p[:,:,:,1]
    edema = p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(28, 60))
    f, axarr = plt.subplots(1,5, figsize = (28, 60)) 
    for i in range(5): # for each image, add brain background
        axarr[i].imshow(cv2.resize(original[:,:,start_slice+Volume_starts_at], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
    axarr[0].imshow(cv2.resize(original[:,:,start_slice+Volume_starts_at], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    axarr[1].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[1].title.set_text('all classes')
    axarr[2].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[2].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[3].imshow(core[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[4].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    # plt.show()
    st.pyplot()
# Streamlit App
def main():
    menu = ["Home" ,"Project report", "Model Demo","About Me"]
    app_mode = st.sidebar.radio("", menu)
    if app_mode == "About Me":
        st.title("About Me")
        st.write("👋 Hello! I'm Hemang Joshi a passionate B.Tech CSE student with a deep enthusiasm for AI, machine learning, and computer vision. My project, TumorDetect 3D, leverages cutting-edge UNET architecture neural networks to perform 3D brain tumor segmentation from MRI images. Beyond the digital world, you'll find me exploring the latest AI advancements and gazing at the stars in the night sky. When not coding, I unwind through cycling and swimming. 🚀🧠 ")
        st.write("### Connect with me:")
        st.write("[LinkedIn](https://www.linkedin.com/in/hemang-joshi-2030/)")
        st.write("[Github](https://github.com/HemangCodesAI)")
    elif app_mode == "Model Demo":
        st.title("MRI Segmentation Results")
        # Upload an image
        uploaded_flair =st.file_uploader(f"### Upload a flair.nii file", type=["nii", "nii.gz"])
        uploaded_t1ce = st.file_uploader(f"### Upload a t1ce.nii file", type=["nii", "nii.gz"])

        if uploaded_flair and uploaded_t1ce is not None:
            with tempfile.NamedTemporaryFile(delete=False,suffix=".nii") as temp_file:
                    temp_file.write(uploaded_flair.read())
                    temp_file_path = temp_file.name
            flair = nib.load(temp_file_path,mmap=False).get_fdata()
            print(flair.shape)
            with tempfile.NamedTemporaryFile(delete=False,suffix=".nii") as temp_file:
                    temp_file.write(uploaded_t1ce.read())
                    temp_file_path = temp_file.name
            t1ce = nib.load(temp_file_path,mmap=False).get_fdata()
            print(t1ce.shape)  


            # Process the uploaded image
            data=process_data(flair,t1ce) 
            
            # button for prediction
            if st.button("Predict"):
                prediction = predict(data)
                show_output_in_streamlit1(flair,prediction)


        # Display the result
    elif app_mode == "Project report":
        # Title
        st.title('Brain Tumor Segmentation using MRI Images')

        # Introduction
        st.header('Introduction:')
        st.write("The project aimed to create a model for segmenting brain tumors into three categories (Necrotic/Core, Edema, Enhancing) from MRI images. The model was developed using a combination of deep learning techniques and medical imaging data from Kaggle. The core technologies used in this project include Python, Keras, TensorFlow, and Nilearn.")

        # Methodology
        st.header('Methodology:')
        st.subheader('Data Preparation:')
        st.write("- MRI images were obtained from the BraTS 2020 dataset on Kaggle, which included images in various modalities such as FLAIR, T1, T1ce, T2, and segmentation masks.")
        st.image(Image.open("tumor.png"), caption="Sample Slices", use_column_width=True)

        st.write("- Data preprocessing involved resizing the images to a uniform size (128x128) for consistency.")

        st.subheader('Model Architecture:')
        st.write("- A U-Net architecture was employed for brain tumor segmentation, known for its success in medical image segmentation tasks.")
        st.write("- The model consists of an encoder-decoder structure with skip connections to capture both low-level and high-level features.")
        st.write("- The final layer uses softmax activation to classify pixels into one of the three tumor categories.")
        st.image(Image.open("u-net-architecture.png"), caption="U-NET Architecture", use_column_width=True)
        st.subheader('Loss Functions:')
        st.write("- The model was trained using categorical cross-entropy as the loss function for multiclass segmentation.")
        st.write("- Additional evaluation metrics included Dice Coefficient, Precision, Sensitivity, Specificity, and Mean Intersection over Union (IoU) to assess segmentation performance.")

        # Results
        st.header('Results:')
        st.write("- The model was trained and evaluated using a dataset split into training, validation, and test sets.")
        st.write("- Performance metrics were monitored throughout training, and the model was optimized using techniques like early stopping and learning rate reduction.")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history_data['epoch'], history_data['accuracy'], label='Training Accuracy', marker='o')
        ax.plot(history_data['epoch'], history_data['val_accuracy'], label='Validation Accuracy', marker='o')
        ax.plot(history_data['epoch'], history_data['loss'], label='Training Loss', marker='o')
        ax.plot(history_data['epoch'], history_data['val_loss'], label='Validation Loss', marker='o')
        ax.plot(history_data['epoch'], history_data['dice_coef'], label='Dice Coefficient', marker='o')
        ax.plot(history_data['epoch'], history_data['mean_io_u'], label='Mean IoU', marker='o')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metrics")
        ax.legend()
        st.pyplot(fig)
        st.write("- Validation results indicated strong segmentation performance, with high values for accuracy, Dice Coefficient, and IoU.")
        st.write("- The model was evaluated on the test dataset, demonstrating its ability to generalize to unseen data.")
        st.image(Image.open("result 1.png"), caption="Sample Predections", use_column_width=True)
        st.image(Image.open("result 12png.png"), caption="Sample Predections", use_column_width=True)

        # Discussion
        st.header('Discussion:')
        st.write("- The U-Net architecture showed promise in accurately segmenting brain tumors from MRI images.")
        st.write("- High values of Dice Coefficient and IoU indicate good agreement between predicted and ground truth segmentations.")
        st.write("- The model's generalization to unseen data is crucial for its clinical utility.")
        st.write("- Further fine-tuning and augmentation techniques could potentially improve performance, especially in handling variations in MRI data.")

        # Conclusion
        st.header('Conclusion:')
        st.write("The developed model provides a valuable tool for automated brain tumor segmentation from MRI images. Accurate segmentation of brain tumors is vital for diagnosis and treatment planning in clinical settings. Future work may involve deploying this model in clinical environments and fine-tuning it for specific applications.")
        st.write("This project demonstrates the potential of deep learning in medical image analysis and its positive impact on healthcare.")
    elif app_mode == "Home":
        st.title("Brain Tumor Segmentation using MRI Images")
        st.image(Image.open("home.jpeg"), caption="Sample Slices", use_column_width=True)

 
    

if __name__ == "__main__":
    main()
