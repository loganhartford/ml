import gradio as gr
from fastai.vision.all import *
import skimage
print(gr.__version__)

def label_func(f): return f[0].isupper()

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Cat Detector"
description = "A cat vs dog classifier trained on the Oxford Pets dataset with fastai."
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"

def get_relative_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Construct the relative path
            relative_path = os.path.join(root, file)
            file_paths.append(relative_path)
    return file_paths
examples = get_relative_file_paths("example_imgs")

interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch()

