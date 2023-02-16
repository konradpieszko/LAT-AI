# This is a code demo code for te LAT-AI-reduced model. This is for testing purposes only. The paper is currently undergoing revision
# Author: Konrad Pieszko, 2023
# Description: LAT-AI reduced model is design to select patients on chronic oral anticoagulation, who have low risk of left atrial appendage thrombus (LAT)


import gradio as gr
import pickle
import os





def make_prediction(age, arrythmia_duration_radio,TEE_base_rythm_radio,HF_status_radio, NYHA_radio,EF, LA_dimension,LAVI,):
    with open("LATAI_reduced_model.pickle", "rb") as f:
        clf  = pickle.load(f)
        print(arrythmia_duration_radio)
        arrythmia_duration_dict={"Paroxysmal":2, "Persistent":1, "Long-standing peristent":1}
        arrythmia_duration=arrythmia_duration_dict[arrythmia_duration_radio]

        NYHA_dict = {"I-II":1, "III":2, "IV":3, "No HF":4}
        NYHA = NYHA_dict[NYHA_radio]

        TEE_base_rythm_dict={"Sinus rhythm":0,"Atrial fibrillation":1,"Atrial flutter":2}
        TEE_base_rythm = TEE_base_rythm_dict[TEE_base_rythm_radio]

        HF_dict= {"HFrEF":3, "HFmrEF":1 ,"HFpEF":2 ,"No HF":4}
        HF_status =HF_dict[HF_status_radio]

        preds = clf.predict_proba([[age, arrythmia_duration,LA_dimension,LAVI,NYHA,HF_status,TEE_base_rythm,EF]])[:,1]
    if preds <0.32:
            return f"Predicted score: {preds}, proceed without TOE"
    return f"Predicted score: {preds}, perform TOE"



experiment_name = "lattee_xgboost_resulst_final_rev"

#Create the input component for Gradio since we are expecting 4 inputs

age_in = gr.Slider(label = "Enter the Age of the Individual",value=55)

arrythmia_duration_in = gr.Radio(["Paroxysmal", "Persistent", "Long-standing peristent"], label="Arrhythmia type", value="Persistent")
TEE_base_rythm_in = gr.Radio(["Sinus rhythm","Atrial fibrillation","Atrial flutter"],label = "Rhytm at the moment:", value="Atrial fibrillation")
#labile_INR_in = gr.Checkbox(label = "Labile INR. For patients on Vitamin K antagonists, check below if less then 60% of available INR is within therapeutic range:")
HF_status_in = gr.Radio(["HFrEF", "HFmrEF" ,"HFpEF" ,"No HF"],label = "Heart failure:", value="No HF")
NYHA_in = gr.Radio(["I-II", "III", "IV", "No HF"], label = "NYHA class", value="I-II")
EF_in = gr.Slider(label = "Left ventricular Ejection Fraction", value=50)

LA_dimension_in = gr.Slider(label = "Left atrium anteroposterior dimension in parasternal short-axis view [mm]", value=40)
LAVI_in = gr.Slider(label = "LAVI (left atrial volume indexed to body surface area in ml/m2 (if available)", value=20)
#APTT_in = gr.Slider(label= "APTT (activated partial thromboplastin time) [seconds]", value=36)

#LA_surface_in = gr.Slider(label = "Left atrial area in sqare cm (if available)", value=20)



# We create the output
output = gr.Textbox()


app = gr.Interface(fn = make_prediction, inputs=[age_in,
 arrythmia_duration_in,TEE_base_rythm_in,HF_status_in,NYHA_in,EF_in,LA_dimension_in,LAVI_in], outputs=output,css="body {font-size: 100%;}")


app.launch(share=False)
