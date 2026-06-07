import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# =====================
# 1. DATASET
# =====================
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print("Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f"Training images: {X_train.shape}")
print(f"Test images: {X_test.shape}")

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# =====================
# 2. CNN MODEL
# =====================
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# =====================
# 3. TRAIN
# =====================
print("\nTraining model...")
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# =====================
# 4. EVALUATE
# =====================
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

# =====================
# 5. GRAPHS
# =====================
fig = plt.figure(figsize=(20, 15))
fig.suptitle('Image Classifier — CNN on CIFAR-10', fontsize=18, fontweight='bold')

gs = gridspec.GridSpec(3, 3, figure=fig)

# Graph 1 - Sample Images
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
ax1.set_title('Sample Training Images', fontsize=14)
for i in range(10):
    ax = fig.add_axes([0.05 + i*0.09, 0.72, 0.08, 0.1])
    ax.imshow(X_train[i])
    ax.set_title(class_names[y_train[i][0]], fontsize=8)
    ax.axis('off')

# Graph 2 - Training Accuracy
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(history.history['accuracy'], label='Train', color='blue')
ax2.plot(history.history['val_accuracy'], label='Val', color='orange')
ax2.set_title('Model Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Graph 3 - Training Loss
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(history.history['loss'], label='Train', color='blue')
ax3.plot(history.history['val_loss'], label='Val', color='orange')
ax3.set_title('Model Loss')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Graph 4 - Confusion Matrix
ax4 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=False, cmap='Blues', ax=ax4,
            xticklabels=class_names, yticklabels=class_names)
ax4.set_title('Confusion Matrix')
ax4.tick_params(axis='x', rotation=45)

# Graph 5 - Per Class Accuracy
ax5 = fig.add_subplot(gs[2, :])
class_acc = cm.diagonal() / cm.sum(axis=1) * 100
bars = ax5.bar(class_names, class_acc, color='steelblue', edgecolor='black')
ax5.set_title('Per Class Accuracy')
ax5.set_ylabel('Accuracy %')
ax5.set_ylim(0, 100)
for bar, acc in zip(bars, class_acc):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('image_classifier_results.png', dpi=100, bbox_inches='tight')
plt.show()
print("Graph saved!")

# =====================
# 6. PREDICT SINGLE IMAGE
# =====================
print("\n=== Single Image Prediction ===")
idx = np.random.randint(0, len(X_test))
img = X_test[idx]
true_label = class_names[y_test[idx][0]]

pred = model.predict(img.reshape(1, 32, 32, 3), verbose=0)
pred_label = class_names[np.argmax(pred)]
confidence = np.max(pred) * 100

print(f"True Label: {true_label}")
print(f"Predicted: {pred_label}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(f"True: {true_label} | Pred: {pred_label} ({confidence:.1f}%)")
plt.axis('off')
plt.savefig('single_prediction.png')
plt.show()

print("\nDay 6 Complete!")