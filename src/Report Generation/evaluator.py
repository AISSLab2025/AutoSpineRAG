
import evaluate
from evaluate import load
from bert_score import score
import pandas as pd
from transformers import AutoTokenizer
import os

# change the model to dynamic

bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')
bertscore = load("bertscore")
frugalscore = evaluate.load("frugalscore")

bert_model_type = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(bert_model_type)

def truncate_text(text, max_len=512):
    tokens = tokenizer.encode(text, max_length=max_len, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def Evaluate(folder_path):
    llm_name = os.path.basename(folder_path)
    df = pd.read_excel(f'{folder_path}/{llm_name}_patient_predictions_original_reports_merged.xlsx')
    # --- Initialize lists for evaluation results ---
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeLsum_scores = []
    bertscore_precision = []
    bertscore_recall = []
    bertscore_f1 = []
    clinical_bertscore_precision = []
    clinical_bertscore_recall = []
    clinical_bertscore_f1 = []
    meteor_scores = []
    frugal_score = []
    blue_score = []
    blue1_scores = []
    blue2_scores = []
    blue3_scores = []
    blue4_scores = []
    processed_indices = []
    
    # --- Iterate through each row of the DataFrame ---
    for index, row in df.iterrows():
        # Validate: Check that both the generated Findings and Clinician_Notes exist and are non-empty strings.
        if pd.isna(row['predicted_notes']) or pd.isna(row['clinician_notes']):
            continue
        if not isinstance(row['predicted_notes'], str) or not isinstance(row['clinician_notes'], str):
            continue
        if row['predicted_notes'].strip() == "" or row['clinician_notes'].strip() == "":
            continue
        
        # In this evaluation, predictions come from the generated Findings and references are the Clinician_Notes.
        predictions = [row['predicted_notes']]
        references = [row['clinician_notes']]
        
        try:
            # --- Compute METEOR ---
            meteor_result = meteor.compute(predictions=predictions, references=references)
            meteor_scores.append(meteor_result['meteor'])
    
            # --- Compute Frugal Score ---
            frugalscore_result = frugalscore.compute(predictions=predictions, references=references)
            frugal_score.append(frugalscore_result['scores'][0])

             # --- Compute ROUGE ---
            rouge_result = rouge.compute(predictions=predictions, references=references)
            rouge1_scores.append(rouge_result['rouge1'])
            rouge2_scores.append(rouge_result['rouge2'])
            rougeL_scores.append(rouge_result['rougeL'])
            rougeLsum_scores.append(rouge_result['rougeLsum'])
            
            # --- Compute BLEU ---
            bleu_result = bleu.compute(predictions=predictions, references=references)
            blue_score.append(bleu_result['bleu'])
            # Assuming bleu_result['precisions'] is a list of scores for n-grams 1 to 4:
            blue1_scores.append(bleu_result['precisions'][0])
            blue2_scores.append(bleu_result['precisions'][1])
            blue3_scores.append(bleu_result['precisions'][2])
            blue4_scores.append(bleu_result['precisions'][3])
            
            # --- Compute BERTScore ---
            bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
            bertscore_precision.append(bertscore_result['precision'][0])
            bertscore_recall.append(bertscore_result['recall'][0])
            bertscore_f1.append(bertscore_result['f1'][0])
            
            
            # truncate long predictions
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                pred_len = len(tokenizer.encode(pred))
                ref_len = len(tokenizer.encode(ref))
                if pred_len > 512 or ref_len > 512:
                    print(f"Row {i} too long: pred_len={pred_len}, ref_len={ref_len}")
                    
            predictions = [truncate_text(p) for p in predictions]

            # --- Compute Clinical BERT Score
            P, R, F1 = score(predictions, references, model_type=bert_model_type, num_layers=12, lang="en", verbose=True)
            clinical_bertscore_precision.append(P.mean().item())
            clinical_bertscore_recall.append(R.mean().item())
            clinical_bertscore_f1.append(F1.mean().item())
        
            # Record the index for which all metrics were computed successfully.
            processed_indices.append(index)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    # --- Create a new DataFrame with the results (using Patient_ID from the merged CSV) ---
    results_df = pd.DataFrame({
        'image': df.loc[processed_indices, 'patient_ID'].values,
        'rouge1': rouge1_scores,
        'rouge2': rouge2_scores,
        'rougeL': rougeL_scores,
        'rougeLsum': rougeLsum_scores,
        'bertscore_precision': bertscore_precision,
        'bertscore_recall': bertscore_recall,
        'bertscore_f1': bertscore_f1,
        'clinical_bertscore_precision': clinical_bertscore_precision,
        'clinical_bertscore_recall': clinical_bertscore_recall,
        'clinical_bertscore_f1': clinical_bertscore_f1,
        'meteor': meteor_scores,
        'frugal': frugal_score,
        'blue': blue_score,
        'blue1': blue1_scores,
        'blue2': blue2_scores,
        'blue3': blue3_scores,
        'blue4': blue4_scores
    })
    
    # --- Save the evaluation results to a new CSV file ---
    output_file = f'{folder_path}_report_evaluation.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")
    
def create_evaluation_data(folder_path):
    llm_name = os.path.basename(folder_path)
    testing_predict_reports_df = pd.read_excel(f'{folder_path}/{llm_name}_predictions.xlsx')
    input_file = 'src/RadiologistsReport.xlsx'
    df = pd.read_excel(input_file)

    directory_path = 'mendeley_dicom_testing/dicom'

    # List all files and remove leading zeros from each filename (excluding extension)
    file_names = [
        str(int(os.path.splitext(file)[0]))  # Convert to int to remove leading zeros, then back to str
        for file in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, file))
    ]
    training_data_reports = []
    testing_data_reports = []
    
    for _, row in df.iterrows():
        patient_id = str(row['Patient ID']).lstrip('0') or '0'  # Normalize to match the format in file_names
        report = {
            "patient_ID": row['Patient ID'],
            "clinician_notes": row["Clinician's Notes"]
        }
        
        if patient_id in file_names:
            testing_data_reports.append(report)
        else:
            training_data_reports.append(report)

    print("Training Data Reports: ", len(training_data_reports))
    print("Testing Data Reports: ", len(testing_data_reports))
    testing_data_reports_df = pd.DataFrame(testing_data_reports)

    merged_df = pd.merge(testing_data_reports_df, testing_predict_reports_df, on="patient_ID", how="left")
    merged_df.to_excel(f'{folder_path}/{llm_name}_patient_predictions_original_reports_merged.xlsx', index=False)
    return merged_df

# llm_names = ["deepseek-r1:7b"]
evaluation_folder = "evaluation_results_ablation_parts\evaluation_results_ablation_parts_RAG_agentic_writing_style"
# llm_names = ["gemma3:12b", "mistral:7b", "llama3.1:8b", "phi4:14b", "qwen3:8b"]
llm_names = ["gemma3:12b"]
for llm_name in llm_names:
    # evaluation_folder = "evaluation_results_RAG_agentic"

    safe_path_name = llm_name.replace(":", "_").replace("/", "_").replace("-", "_").replace(".", "_")

    merged_df = create_evaluation_data(os.path.join(evaluation_folder, safe_path_name))
    
import os
# llm_names = ["deepseek-r1:7b"]
# llm_names = ["phi4", "gemma3:12b", "qwen2.5:14b"]

for llm_name in llm_names:
    # evaluation_folder = "evaluation_results_RAG_agentic"

    safe_path_name = llm_name.replace(":", "_").replace("/", "_").replace("-", "_").replace(".", "_")
    Evaluate(os.path.join(evaluation_folder, safe_path_name))
    