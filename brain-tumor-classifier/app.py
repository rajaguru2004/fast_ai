import gradio as gr
from fastai.vision.all import *

# Load trained model
learn = load_learner("brain_tumor_classifier.pkl")

classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

def predict(image):
    pred, pred_idx, probs = learn.predict(PILImage.create(image))
    
    pred_class = classes[pred_idx]
    confidence = f"{probs[pred_idx]*100:.2f}%"
    prob_dict = {classes[i]: float(probs[i]) for i in range(4)}
    
    return pred_class, confidence, prob_dict

# JavaScript to redirect URL and force light mode theme parameter
js_force_light = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.href);
    }
}
"""

# Bulletproof CSS override to map dark mode variables to light mode colors
css_force_light = """
body, html, .gradio-container {
    background-color: #f9fafb !important;
}
.dark {
    --background-fill-primary: #f9fafb !important;
    --background-fill-secondary: #f3f4f6 !important;
    --block-background-fill: #ffffff !important;
    --body-text-color: #111827 !important;
    --body-text-color-subdued: #4b5563 !important;
    --block-title-text-color: #1f2937 !important;
    --block-label-text-color: #374151 !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #e5e7eb !important;
    --button-secondary-background-fill: #f3f4f6 !important;
    --button-secondary-text-color: #1f2937 !important;
    --border-color-primary: #e5e7eb !important;
}
"""

# Build customized UI using Blocks for a premium layout
with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 600px; margin: 0 auto; padding: 20px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #1e1b4b; margin-bottom: 5px;">
                Brain Tumor MRI Classifier
            </h1>
            <p style="font-size: 1rem; color: #4b5563; margin-bottom: 20px;">
                Upload a Brain MRI image to classify the tumor type.
            </p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload MRI Image")
            btn_submit = gr.Button("Analyze Scan", variant="primary")
            
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Results")
            with gr.Row():
                txt_pred = gr.Textbox(label="Prediction", interactive=False)
                txt_conf = gr.Textbox(label="Confidence", interactive=False)
            
            lbl_probs = gr.Label(num_top_classes=4, label="Class Probabilities")
            
    btn_submit.click(
        fn=predict, 
        inputs=img_input, 
        outputs=[txt_pred, txt_conf, lbl_probs]
    )
    
    gr.Examples(
        examples=[
            "assets/glioma_sample.jpg",
            "assets/meningioma_sample.jpg",
            "assets/notumor_sample.jpg",
            "assets/pituitary_sample.jpg"
        ],
        inputs=img_input,
        label="Example Scans"
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        js=js_force_light,
        css=css_force_light
    )
