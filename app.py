import gradio as gr
from src.pipeline.prediction_pipline import CustomData, PredictPipeline

# Constants
CUT_CHOICES = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_CHOICES = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY_CHOICES = ["I1", "SI2", "SI2", "VS2", "VS1", "VVS2", "VVS1", "IF"]

def predict(inp_carat, inp_depth, inp_table, inp_x, inp_y, inp_z, selected_cut,selected_color,selected_clarity):
    data = CustomData(
        carat=inp_carat,
        depth=inp_depth,
        table=inp_table,
        x=inp_x,
        y=inp_y,
        z=inp_z,
        cut=selected_cut,
        color=selected_color,
        clarity=selected_clarity       
    )
    
    parsed_data = data.get_data_as_dataframe()
    prediction_pipline = PredictPipeline()
    prediction = prediction_pipline.predict(parsed_data)
    
    return round(prediction[0], 2)
    

with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Diamond Price Prediction.")
    gr.Markdown("### For predicting the value enter the inputs and then click on **Predict** to see the result.")
    
    #with gr.Row():
        
    with gr.Row():
        carat = gr.Number(label="Carat")
        depth = gr.Number(label="Depth")
        table = gr.Number(label="Table")
        
    with gr.Row():
        x = gr.Number(label="X")
        y = gr.Number(label="Y")
        z = gr.Number(label="Z")
        
    with gr.Row():
        cut = gr.Dropdown(label="Cut", choices=CUT_CHOICES, 
            info="Diamond cut specifically refers to the quality of a diamond's angles, proportions, symmetrical facets, brilliance, fire, scintillation and finishing details.")
        color = gr.Dropdown(label="Color", choices=COLOR_CHOICES,
            info="Diamond color is graded in terms of how white or colorless a diamond is.")
        clarity = gr.Dropdown(label="Clarity", choices=CLARITY_CHOICES,
            info="A diamond's clarity grade evaluates how clean a diamond is from both inclusions and blemishes.")
        
    gr.Markdown("### Prediction: ")
    result = gr.Textbox(label="Predicted Price")
    predict_btn = gr.Button("Predict")
    predict_btn.click(fn=predict, inputs=[carat,depth,table,x,y,z,cut,color,clarity], outputs=result)
    

demo.launch()