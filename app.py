import gradio as gr
from fastai.vision.all import *
from os.path import dirname, realpath, join
import pathlib

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner('./all_resnet50_1.pkl')

labels = [
    'This is Hass, and it will take about 5-7 days to ripe', 
    'This is Hass, and it will take about 3-4 days to ripe', 
    'This is Hass, and it is ready to eat', 
    'This is Mayuang, and it will take about 4-6 days to ripe', 
    'This is Mayuang, and it will take about 2-3 days to ripe', 
    'This is Mayuang, and it will take about 1-2 days to ripe', 
    'This is Mayuang, and it is ready to eat']

# def clear():
#   audio_input.clear()

# clear_btn = gr.Button(value="Clear")
# clear_btn.click(clear, [], [])


def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr_interface = gr.Interface(fn=predict, 
                            inputs=gr.inputs.Image(shape=(224, 224)),
                            outputs=gr.outputs.Label(num_top_classes=3), 
                            title="AvocadoRipenessPredictor",
                            description="* The least ripe side of the avocado should be taken.", 
                            interpretation="default",
                            examples=[
                                ["./statics/t_03a.jpg"],
                                ["./statics/t_08b.jpg"],
                                ["./statics/th_06b.jpg"],
                            ],
                            # clean_command=True  # Add the clean button
                            )
gr_interface.launch()
