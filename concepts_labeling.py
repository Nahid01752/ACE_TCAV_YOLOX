import os
import re
import argparse
from collections import Counter
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline

def process_line(line):
    pattern = re.compile(r'^(.*concept\d+)_USER:.*ASSISTANT: (.*)$')
    match = pattern.match(line)
    if match:
        prefix = match.group(1)
        label_info = match.group(2)
        updated_line = f"{prefix}_{label_info}"
        return updated_line
    else:
        print("cannot be processed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="USER: <image>\nLabel this image within 3 words. \nASSISTANT:", help='Prompt to be used for labeling')
    parser.add_argument('--tcav_file_path', type=str, default="/workspace/ACE_TCAV_Pytorch/outputs/test/results_summaries/TCAV_ace_results.txt", help='Path to the TCAV results file')
    parser.add_argument('--root_dir', type=str, default="/workspace/ACE_TCAV_Pytorch/outputs/test/concepts", help='Root directory containing concept folders')
    args = parser.parse_args()

    file_p = '/workspace/Llava/concept_labels.txt'
    TCAV_file_path = args.tcav_file_path
    prompt = args.prompt
    root_dir = args.root_dir

    if os.path.exists(file_p):
        os.remove(file_p)
        print("existing concept_labels.txt has been removed")
    else:
        print("concept_labels.txt doesn't exist")

    # Preparing the quantization config to load the model in 4bit precision
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model_id = "llava-hf/llava-1.5-7b-hf"
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

    output_file_path = file_p

    # Regex to match folders ending with 'concept' followed by an integer
    folder_regex = re.compile(r'concept\d+$')
    second_concepts = {}  # dictionary to hold concept labels

    # Iterate over all the concept folders in the root directory
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path) and folder_regex.search(folder_name):
            labels = []
            for image_name in os.listdir(folder_path):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, image_name)
                    image = Image.open(image_path).convert('RGB')
                    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
                    label = outputs[0]["generated_text"].strip()
                    labels.append(label)
            # find the most common label
            if labels:
                most_common_label, count = Counter(labels).most_common(1)[0]
                percentage = count / len(labels)
                output = f"{folder_name}_{most_common_label}_{percentage}\n"
                #output = f"{folder_name}_{most_common_label}_{count}_{len(labels)}_{percentage}\n"
                formatted_output = ' '.join(output.split())
                print(formatted_output)
                processed_line = process_line(formatted_output)
                parts = processed_line.strip().split('_')
                concept_key = parts[3]
                extra_info = '_'.join(parts[4:])
                second_concepts[concept_key] = extra_info

    with open(TCAV_file_path, 'r') as file:
        first_lines = file.readlines()

    # dictionary to hold TCAV concepts
    first_concepts = {}
    for line in first_lines:
        if 'concept' in line:
            concept_key = line.split(':')[1]
            first_concepts[concept_key] = line.strip()

    updated_lines = []
    for concept, line in first_concepts.items():
        concept_number = concept.split('concept')[1]
        for second_key, extra_info in second_concepts.items():
            second_number = second_key.split('concept')[1]
            if concept_number == second_number:
                parts = line.split(':')
                updated_line = f"{parts[0]}:{parts[1]} {extra_info}:{parts[2]}"
                updated_lines.append(updated_line)
                break
        else:
            updated_lines.append(line)

    with open(output_file_path, 'w') as file:
        file.write('\n')
        file.write('\t\t\t---TCAV scores---\n\n')
        for line in updated_lines:
            file.write(line + '\n')
        file.write('\n')
        file.write(prompt + '\n')

    print("Labeling Done")

if __name__ == "__main__":
    main()
