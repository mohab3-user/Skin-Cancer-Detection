import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.fft import fft2, fftshift
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════
#   STEP 1 — OPEN AND RESIZE ONE IMAGE
# ══════════════════════════════════════════

def open_one_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (64, 64))
    return image


# ══════════════════════════════════════════
#   STEP 2 — LOAD ALL IMAGES FROM A FOLDER
# ══════════════════════════════════════════

def load_all_images(folder_path, label, limit=100):
    all_images = []
    all_labels = []

    file_list = os.listdir(folder_path)

    for file_name in file_list[:limit]:
        full_path = os.path.join(folder_path, file_name)
        image     = open_one_image(full_path)

        if image is not None:
            all_images.append(image)
            all_labels.append(label)

    return all_images, all_labels


# ══════════════════════════════════════════
#   STEP 3 — DSP: CLEAN THE IMAGE (FILTER)
# ══════════════════════════════════════════

def clean_image(image):
    gray_image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cleaned_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return cleaned_image


# ══════════════════════════════════════════
#   STEP 4 — DSP: FOURIER TRANSFORM
# ══════════════════════════════════════════

def apply_fourier(cleaned_image):
    fourier_result = fft2(cleaned_image)
    centered       = fftshift(fourier_result)
    frequencies    = np.abs(centered)
    return frequencies


# ══════════════════════════════════════════
#   STEP 5 — DSP: DETECT EDGES (SOBEL)
# ══════════════════════════════════════════

def detect_edges(cleaned_image):
    horizontal_edges = cv2.Sobel(cleaned_image, cv2.CV_64F, 1, 0, ksize=3)
    vertical_edges   = cv2.Sobel(cleaned_image, cv2.CV_64F, 0, 1, ksize=3)
    all_edges        = np.sqrt(horizontal_edges**2 + vertical_edges**2)
    return all_edges


# ══════════════════════════════════════════
#   STEP 6 — EXTRACT 12 NUMBERS FROM IMAGE
# ══════════════════════════════════════════

def get_12_numbers(image):
    cleaned     = clean_image(image)
    frequencies = apply_fourier(cleaned)
    edges       = detect_edges(cleaned)
    hsv_image   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    number_1  = np.mean(frequencies)
    number_2  = np.std(frequencies)
    number_3  = np.max(frequencies)
    number_4  = np.var(cleaned)
    number_5  = np.mean(edges)
    number_6  = np.std(edges)
    number_7  = np.mean(hsv_image[:, :, 0])
    number_8  = np.mean(hsv_image[:, :, 1])
    number_9  = np.mean(hsv_image[:, :, 2])
    number_10 = np.std(hsv_image[:, :, 0])
    number_11 = np.std(hsv_image[:, :, 1])
    number_12 = np.std(hsv_image[:, :, 2])

    return [number_1, number_2, number_3, number_4, number_5, number_6,
            number_7, number_8, number_9, number_10, number_11, number_12]


# ══════════════════════════════════════════
#   STEP 7 — LOAD THE DATASET
# ══════════════════════════════════════════

print("Loading benign images...")
benign_images, benign_labels = load_all_images('data/train/benign', label=0)

print("Loading malignant images...")
malignant_images, malignant_labels = load_all_images('data/train/malignant', label=1)

all_images = benign_images + malignant_images
all_labels = benign_labels + malignant_labels

total_images    = len(all_images)
total_benign    = len(benign_images)
total_malignant = len(malignant_images)

print(f"Done! Loaded {total_images} images total")


# ══════════════════════════════════════════
#   STEP 8 — EXTRACT NUMBERS FROM ALL IMAGES
# ══════════════════════════════════════════

print("Extracting features from every image...")

all_numbers = []
for image in all_images:
    numbers = get_12_numbers(image)
    all_numbers.append(numbers)

all_numbers = np.array(all_numbers)
all_labels  = np.array(all_labels)

print("Done extracting features!")


# ══════════════════════════════════════════
#   STEP 9 — NORMALIZE THE NUMBERS
# ══════════════════════════════════════════

normalizer         = StandardScaler()
normalized_numbers = normalizer.fit_transform(all_numbers)


# ══════════════════════════════════════════
#   STEP 10 — SPLIT INTO TRAIN AND TEST
# ══════════════════════════════════════════

train_numbers, test_numbers, train_labels, test_labels = train_test_split(
    normalized_numbers, all_labels,
    test_size=0.2,
    random_state=42
)


# ══════════════════════════════════════════
#   STEP 11 — TRAIN THE MODEL
# ══════════════════════════════════════════

print("Training the model...")

model = SVC(kernel='rbf', probability=True)
model.fit(train_numbers, train_labels)

print("Model trained!")


# ══════════════════════════════════════════
#   STEP 12 — TEST THE MODEL
# ══════════════════════════════════════════

predicted_labels = model.predict(test_numbers)

accuracy  = accuracy_score(test_labels, predicted_labels)
correct   = int(np.sum(predicted_labels == test_labels))
incorrect = int(np.sum(predicted_labels != test_labels))
tested_on = len(test_labels)

print(f"Accuracy: {accuracy*100:.1f}%  ({correct} correct, {incorrect} wrong out of {tested_on})")


# ══════════════════════════════════════════
#   STEP 13 — PICK A RANDOM IMAGE TO SHOW
# ══════════════════════════════════════════

random_index = np.random.randint(0, total_images)
random_image = all_images[random_index]

cleaned      = clean_image(random_image)
frequencies  = np.log(np.abs(fftshift(fft2(cleaned))) + 1)
edges        = detect_edges(cleaned)
original_rgb = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════
#   STEP 14 — PREDICT THAT RANDOM IMAGE
# ══════════════════════════════════════════

image_numbers      = get_12_numbers(random_image)
image_normalized   = normalizer.transform([image_numbers])

prediction         = model.predict(image_normalized)[0]
prediction_chances = model.predict_proba(image_normalized)[0]
confidence         = max(prediction_chances) * 100

is_cancer   = prediction == 1
result_word = "MALIGNANT" if is_cancer else "BENIGN"
risk_word   = "HIGH RISK" if is_cancer else "LOW RISK"


# ══════════════════════════════════════════
#   STEP 15 — DRAW THE WINDOW
# ══════════════════════════════════════════

ACCENT = "#FF4C4C" if is_cancer else "#00E5A0"
BG     = "#0D0D1A"
CARD   = "#13132B"
WHITE  = "#F0F0FF"
DIM    = "#7777AA"

fig = plt.figure(figsize=(18, 9), facecolor=BG)

gs = gridspec.GridSpec(
    3, 5, figure=fig,
    left=0.03, right=0.97,
    top=0.88,  bottom=0.08,
    wspace=0.35, hspace=0.5
)

fig.text(0.5, 0.96, "SKIN CANCER DETECTION SYSTEM",
         ha='center', fontsize=20, fontweight='bold',
         color=WHITE, fontfamily='monospace')

fig.text(0.5, 0.915,
         f"DSP Project  ·  SVM Classifier  ·  Trained on {total_images} Images  ·  12 Features Extracted",
         ha='center', fontsize=9, color=DIM, fontfamily='monospace')


dsp_panels = [
    (original_rgb, None,      "ORIGINAL IMAGE",       "Raw skin lesion photo"),
    (cleaned,      'gray',    "GAUSSIAN FILTER",      f"Noise removed — std: {np.std(cleaned):.1f}"),
    (frequencies,  'inferno', "FOURIER TRANSFORM",    f"Freq mean: {np.mean(frequencies):.2f}"),
    (edges,        'gray',    "SOBEL EDGE DETECTION", f"Edge strength: {np.mean(edges):.1f}"),
]

for i, (panel_image, colormap, title, subtitle) in enumerate(dsp_panels):
    ax = fig.add_subplot(gs[:2, i])
    ax.set_facecolor(CARD)
    ax.imshow(panel_image, cmap=colormap, aspect='auto')
    ax.set_title(title,    color=ACCENT, fontsize=8, fontweight='bold', fontfamily='monospace', pad=6)
    ax.set_xlabel(subtitle, color=DIM,   fontsize=7.5, fontfamily='monospace')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)


result_panel = fig.add_subplot(gs[:, 4])
result_panel.set_facecolor(CARD)
result_panel.set_xticks([])
result_panel.set_yticks([])
for spine in result_panel.spines.values():
    spine.set_edgecolor(ACCENT)
    spine.set_linewidth(2)

result_panel.text(0.5, 0.94, "DIAGNOSIS",          ha='center', fontsize=9,  color=DIM,    fontfamily='monospace', transform=result_panel.transAxes)
result_panel.text(0.5, 0.82, result_word,           ha='center', fontsize=22, color=ACCENT, fontweight='bold', fontfamily='monospace', transform=result_panel.transAxes)
result_panel.text(0.5, 0.71, risk_word,             ha='center', fontsize=13, color=WHITE,  fontweight='bold', fontfamily='monospace', transform=result_panel.transAxes)
result_panel.text(0.5, 0.60, "CONFIDENCE",          ha='center', fontsize=8,  color=DIM,    fontfamily='monospace', transform=result_panel.transAxes)
result_panel.text(0.5, 0.50, f"{confidence:.1f}%",  ha='center', fontsize=26, color=WHITE,  fontweight='bold', fontfamily='monospace', transform=result_panel.transAxes)

bar_background = FancyBboxPatch((0.1, 0.40), 0.8, 0.04,                       boxstyle="round,pad=0.01", facecolor='#1E1E3A', edgecolor='none', transform=result_panel.transAxes)
bar_filled     = FancyBboxPatch((0.1, 0.40), 0.8 * (confidence / 100), 0.04,  boxstyle="round,pad=0.01", facecolor=ACCENT,    edgecolor='none', transform=result_panel.transAxes, alpha=0.9)
result_panel.add_patch(bar_background)
result_panel.add_patch(bar_filled)

result_panel.text(0.5, 0.32, "MODEL ACCURACY",                       ha='center', fontsize=8,  color=DIM,   fontfamily='monospace', transform=result_panel.transAxes)
result_panel.text(0.5, 0.23, f"{accuracy*100:.1f}%",                 ha='center', fontsize=20, color=WHITE, fontweight='bold', fontfamily='monospace', transform=result_panel.transAxes)
result_panel.text(0.5, 0.14, f"{correct} correct / {incorrect} wrong", ha='center', fontsize=8, color=DIM,  fontfamily='monospace', transform=result_panel.transAxes)
result_panel.text(0.5, 0.06, f"tested on {tested_on} images",         ha='center', fontsize=7, color=DIM,   fontfamily='monospace', transform=result_panel.transAxes)


bottom_stats = [
    ("TRAINING IMAGES",   str(total_images)),
    ("BENIGN TRAINED",    str(total_benign)),
    ("MALIGNANT TRAINED", str(total_malignant)),
    ("FEATURES USED",     "12"),
]

for i, (stat_label, stat_value) in enumerate(bottom_stats):
    ax = fig.add_subplot(gs[2, i])
    ax.set_facecolor(CARD)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2A4A")
        spine.set_linewidth(1)
    ax.text(0.5, 0.62, stat_value, ha='center', fontsize=18, fontweight='bold', color=ACCENT, fontfamily='monospace', transform=ax.transAxes)
    ax.text(0.5, 0.25, stat_label, ha='center', fontsize=6.5, color=DIM,        fontfamily='monospace', transform=ax.transAxes)


plt.savefig('result.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

print(f"\nResult     : {result_word} — {risk_word}")
print(f"Confidence : {confidence:.1f}%")
print(f"Accuracy   : {accuracy*100:.1f}%")
