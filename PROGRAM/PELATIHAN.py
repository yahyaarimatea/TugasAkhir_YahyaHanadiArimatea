# PELATIHAN

# ==========================================================================
# TUGAS AKHIR
# JUDUL : PENGENALAN NADA PIANIKA MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK
# NAMA  : YAHYA HANADI ARIMATEA
# NIM   : 215114003
# PRODI : TEKNIK ELEKTRO
# ===========================================================================

import numpy as np
import os
from scipy.io import wavfile
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

folder_data = 'E:/DATA'
model_path = 'E:/hasilpelatihan.h5'

label_mapping = {
    'C_tinggi': 7, 'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6
}

label_nama = list(label_mapping.keys())

X = []
y = []

for file in os.listdir(folder_data):
    if file.endswith('.wav'):
        for label in label_mapping:
            if file.startswith(label):
                samp_rate, data = wavfile.read(os.path.join(folder_data, file))
                typ = data.astype(np.float32)
                data = data / np.max(np.abs(data))
                threshold = 0.5
                awal = np.where(data > threshold)[0]
                
                if len(awal) == 0:
                    continue
                data = data[awal[0]:]
                frame_length = 256
                
                if len(data) < frame_length:
                    continue
                frame = data[:frame_length]
                frame = frame / np.max(np.abs(frame))
                hamming = np.hamming(frame_length)
                frame = frame * hamming
                hasil_fft = np.fft.fft(frame)
                magnitude = np.abs(hasil_fft[:frame_length // 2])
                magnitude[0] = 0
                magnitude = magnitude / np.max(magnitude)
                X.append(magnitude)
                y.append(label_mapping[label])
                break

X_train, X_test, y_train, y_test = [], [], [], []

for i in range(8):
    data_i = [x for x, label in zip(X, y) if label == i]
    if len(data_i) >= 30:
        data_i = shuffle(data_i, random_state=42)
        X_train.extend(data_i[:20])
        X_test.extend(data_i[20:30])
        y_train.extend([i]*20)
        y_test.extend([i]*10)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

y_train_cat = to_categorical(y_train, 8)
y_test_cat = to_categorical(y_test, 8)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    Conv1D(2, 3, strides=1, activation='relu', padding='same', input_shape=(128, 1)),
    AveragePooling1D(pool_size=2, strides=2),
    Conv1D(1, 3, strides=1, activation='relu', padding='same'),
    AveragePooling1D(pool_size=4, strides=4),
    Flatten(),
    Dense(8, activation='relu'),
    Dense(8, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_cat, epochs=30, batch_size=2, validation_data=(X_test, y_test_cat))

model.save(model_path)

y_pred = np.argmax(model.predict(X_test), axis=1)

print('\nLaporan Klasifikasi:')
print(classification_report(y_test, y_pred, target_names=label_nama))

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5,4), dpi=100)
ConfusionMatrixDisplay(cm, display_labels=label_nama).plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
plt.tight_layout()

sg.theme('LightGrey1')

layout = [
    [sg.Text(text='Confusion Matrix CNN Pianika',
             font=('Helvetica',16),
             justification='center',
             expand_x=True)],
    [sg.Canvas(key='-CANVAS-')],
    [sg.Button('Keluar')]
]

window = sg.Window('Hasil Training',
                   layout,
                   finalize=True)

canvas_elem = window['-CANVAS-'].TKCanvas
figure_canvas = FigureCanvasTkAgg(fig, canvas_elem)
figure_canvas.draw()
figure_canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, 'Keluar'):
        break

window.close()