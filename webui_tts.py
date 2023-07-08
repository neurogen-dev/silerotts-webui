import gradio as gr
import os

# Создаем список доступных speaker
speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']

def tts(file, speaker):
  # assuming tts.py is in the same directory as this script
  os.system(f"python tts.py --file {file.name} --speaker {speaker}")
  # assuming tts.py outputs a file called output.wav in the same directory
  return "output.wav"

iface = gr.Interface(
  fn=tts,
  inputs=[
        gr.File(label="Ваш txt файл", type="file"),
        gr.Dropdown(label="Выберите диктора", choices=["aidar", "baya", "kseniya", "xenia", "eugene", "random"], value="baya"), # отправляет 
  ],
  outputs=[
        gr.outputs.Audio(type="filepath"),
  ],
  title="TTS by Neurogen",
  description="Загрузите ваш txt файл с текстом и получите на выходе аудиофайл с озвучкой</br>Работает на базе SileroTTS"
)

iface.launch()
