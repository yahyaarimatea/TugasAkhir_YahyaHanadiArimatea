# PENGUJIAN REAL TIME

# ==========================================================================
# TUGAS AKHIR
# JUDUL : PENGENALAN NADA PIANIKA MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK
# NAMA  : YAHYA HANADI ARIMATEA
# NIM   : 215114003
# PRODI : TEKNIK ELEKTRO
# ===========================================================================

import PySimpleGUI as sg
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model, Model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time

model_path = 'E:/hasilpelatihan.h5'
label_mapping = {
    0:'C', 1:'D', 2:'E', 3:'F', 4:'G', 5:'A', 6:'B', 7:'C_tinggi'
}

fig1, ax1 = plt.subplots(figsize=(5, 3.5), dpi=100)
fig1.tight_layout(pad=3.0)
ax1.plot([], [])
ax1.set_xlabel('Data ke')
ax1.set_ylabel('Amplitudo')
ax1.set_xlim(-235, 5035)
ax1.set_ylim(-1, 1)

fig2, ax2 = plt.subplots(figsize=(5, 3.5), dpi=100)
fig2.tight_layout(pad=3.0)
ax2.plot([], [])
ax2.set_xlabel('Data ke')
ax2.set_ylabel('Nilai')
ax2.set_xlim(-1.2, 16.2)
ax2.set_ylim(0, 0.35)

sg.theme('LightGrey1')

layout = [
    [sg.Text(text='Pengujian Nada Alat Musik Pianika Real-Time',
             font=('Helvetica', 18, 'bold'),
             expand_x=True,
             justification='center')],
    [sg.Button('Reset',
               size=(10, 1)),
     sg.Button('Selesai',
               size=(10, 1))],
    [sg.Column([[sg.Text(text='Plot Hasil Rekaman',
                         font=('Helvetica', 12),
                         expand_x=True,
                         justification='center')],
                [sg.Canvas(key='-CANVAS1-')],
                [sg.Text('TIME DOMAIN',
                 font=('Helvetica', 12, 'bold'),
                 justification='center',
                 expand_x=True)]], 
               element_justification='center',
               vertical_alignment='top',
               expand_y=True),

     sg.Column([[sg.Text(text='Plot Input Flatten Layer',
                         font=('Helvetica', 12),
                         expand_x=True,
                         justification='center')],
                [sg.Canvas(key='-CANVAS2-')],
                [sg.Text('FREKUENSI DOMAIN',
                 font=('Helvetica', 12, 'bold'),
                 justification='center',
                 expand_x=True)]],
               element_justification='center',
               vertical_alignment='top',
               expand_y=True),

     sg.Column([[sg.Text(text='Nada Dikenali',
                         font=('Helvetica', 12),
                         expand_x=True,
                         justification='center')],
                [sg.Frame('', [[sg.Text(text='-', key='-NADA-',
                                        font=('Helvetica', 40),
                                        size=(15, 2),
                                        justification='center',
                                        pad=(0, 30))]],
                                        size=(220, 130))]])]]

window = sg.Window('Pengujian Real-Time',
                   layout,
                   finalize=True,
                   element_justification='center',
                   resizable=True)

figure_canvas1 = FigureCanvasTkAgg(fig1, window['-CANVAS1-'].TKCanvas)
figure_canvas1.draw()
figure_canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)
plt.close(fig1)

figure_canvas2 = FigureCanvasTkAgg(fig2, window['-CANVAS2-'].TKCanvas)
figure_canvas2.draw()
figure_canvas2.get_tk_widget().pack(side='top', fill='both', expand=1)
plt.close(fig2)

model = load_model(model_path)
model.summary()
dummy_input = np.zeros((4, 128, 1), dtype=np.float16)
_ = model(dummy_input)
cnn_layer_model = Model(inputs=model.inputs, outputs=model.layers[0].output)
            
flatten_layer_model = Model(inputs=model.inputs, outputs=model.get_layer('flatten').output)

last_process_time = 0
nada_sebelumnya = '-'

while True:
    event, values = window.read(timeout=100)

    if event == sg.WINDOW_CLOSED or event == 'Selesai':
        break

    elif event == 'Reset':
        fig1, ax1 = plt.subplots(figsize=(5, 3.5), dpi=100)
        fig1.tight_layout(pad=3.0)
        ax1.plot([], [])
        ax1.set_xlabel('Data ke')
        ax1.set_ylabel('Amplitudo')
        ax1.set_xlim(-235, 5035)
        ax1.set_ylim(-1, 1)
        figure_canvas1.get_tk_widget().forget()
        figure_canvas1 = FigureCanvasTkAgg(fig1, window['-CANVAS1-'].TKCanvas)
        figure_canvas1.draw()
        figure_canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)
        plt.close(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(5, 3.5), dpi=100)
        fig2.tight_layout(pad=3.0)
        ax2.set_xlabel('Data ke')
        ax2.set_ylabel('Nilai')
        ax2.set_xlim(-1.2, 16.2)
        ax2.set_ylim(0, 0.35)
        figure_canvas2.get_tk_widget().forget()
        figure_canvas2 = FigureCanvasTkAgg(fig2, window['-CANVAS2-'].TKCanvas)
        figure_canvas2.draw()
        figure_canvas2.get_tk_widget().pack(side='top', fill='both', expand=1)
        plt.close(fig2)
        
        window['-NADA-'].update('-')
        continue
    
    fs = 4800
    duration = 1
    if time.time() - last_process_time > 1:
        last_process_time = time.time()
        try:
            recording = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            x = recording[:, 0]

            energy = np.mean(np.square(x))
            if energy < 0.02:
                continue
            
            x0 = x.astype(np.float32)
            x1 = x0 / np.max(np.abs(x0)) if np.max(np.abs(x0)) != 0 else x0

            window_size = 256
            max_energy, start_idx = 0, 0
            for i in range(len(x1) - window_size):
                e = np.sum(x1[i:i+window_size]**2)
                if e > max_energy:
                    max_energy = e
                    start_idx = i

            x2 = x1[start_idx:start_idx+window_size]
            x3 = x2 / np.max(np.abs(x2)) if np.max(np.abs(x2)) != 0 else x2
            x4 = x3 * np.hamming(window_size)
            x5 = np.abs(np.fft.fft(x4)[:window_size // 2])
            x5[0] = 0
            x6 = x5 / np.max(x5) if np.max(x5) != 0 else x5

            x_input = x6[np.newaxis, ..., np.newaxis]

            pred = model.predict(x_input, verbose=0)
            label_idx = np.argmax(pred)
            nada_dikenali = label_mapping[label_idx]

            if nada_dikenali != nada_sebelumnya:
                window['-NADA-'].update(nada_dikenali)
                nada_sebelumnya = nada_dikenali

            ax1.clear()
            ax1.plot(recording, color='blue')
            ax1.set_xlabel('Data ke')
            ax1.set_ylabel('Amplitudo')
            figure_canvas1.draw()

            flatten_output = flatten_layer_model.predict(x_input, verbose=0)[0]
            ax2.clear()
            ax2.stem(np.arange(len(flatten_output)), flatten_output, linefmt='b-', markerfmt='bo', basefmt='k-')
            ax2.set_xlabel('Data ke')
            ax2.set_ylabel('Nilai')
            figure_canvas2.draw()

        except Exception as e:
            sg.popup_error('Error:', str(e))

window.close()