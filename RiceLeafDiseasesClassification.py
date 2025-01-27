# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
%matplotlib inline

# %%
# Pipeline For Image to feature Vector 
import os
from skimage import color
from skimage import transform
labels = ['Bacterial_leaf_blight', 'Brown_spot', 'Leaf_smut']
images_Vector = []
for d in os.listdir('data'):
    for image in os.listdir(f'data\\{d}'):
        #Loading Image
        img = io.imread(f'data\\{d}\\{image}')
        # To Gray
        gray_img = color.rgb2gray(img)
        #Resizing
        resized_img = transform.resize(gray_img, (28,28))
        image_vector = resized_img.ravel()
        # print(image_vector.shape, labels.index(d))
        images_Vector.append([image_vector, labels.index(d)])


# %%
# Featur and target
images_vectors = np.array([image[0] for image in images_Vector])
Labels = np.array([image[1] for image in images_Vector])

# %%
images_vectors[0], Labels[0]

# %%
# Dividing Data into Training and Testing set
from sklearn.model_selection import train_test_split
np.random.seed(42)
train_images, test_images, train_labels, test_labels = train_test_split(
    images_vectors, Labels, test_size=0.25
)

# %%
train_images.shape, train_labels.shape

# %%
# Modeling
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=2000)
model.fit(train_images, train_labels.ravel())

# %%
# Prediction/Validation
ypred = model.predict(test_images)
ypred[0], test_labels[0]

# %%
# Confusion Matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
confusion_matrix(ypred, test_labels)

# %%
plot_confusion_matrix(model, test_images, test_labels.ravel())

# %%
from sklearn.metrics import accuracy_score
accuracy_score(ypred, test_labels)


