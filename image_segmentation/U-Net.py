import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

# ------------------------------
# 1. Load and preprocess image
# ------------------------------
img_path = "D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\peppers.png"
img = Image.open(img_path).convert('RGB')
original_size = img.size  # save original size for resizing later
img = img.resize((572, 572))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # shape (1, 572, 572, 3)

# ------------------------------
# 2. Define / Compile Model
# ------------------------------
model = unet_model(input_shape=(572, 572, 3), num_classes=2)
model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ------------------------------
# 3. Train Model (if not already trained)
# ------------------------------
# train_images, train_masks, val_images, val_masks should be prepared beforehand
model.fit(train_images, train_masks,
          batch_size=4,
          epochs=20,
          validation_data=(val_images, val_masks))

# Save trained weights
model.save_weights("unet_trained_weights.h5")

# ------------------------------
# 4. Load trained weights for prediction
# ------------------------------
model.load_weights("unet_trained_weights.h5")

# ------------------------------
# 5. Predict on new image
# ------------------------------
predictions = model.predict(img_array)  # shape: (1, 572, 572, 2)

# Convert prediction to mask
pred_mask = np.argmax(predictions[0], axis=-1).astype(np.uint8) * 255

# Convert to image and resize to original
pred_mask_img = Image.fromarray(pred_mask)
pred_mask_img = pred_mask_img.resize(original_size)

# Save and show
pred_mask_img.save('predicted_image.jpg')
pred_mask_img.show()
