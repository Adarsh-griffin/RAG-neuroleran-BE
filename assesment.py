import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import textwrap
from langchain.prompts import PromptTemplate



#Load the model
model_path = "C:/techai/program/model"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Automatically chooses GPU if available
    torch_dtype=torch.float16,  # Ensures efficiency
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

#Initialize the pipeline
llm = HuggingFacePipeline(pipeline=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,    # Adjust as needed
))

print("========================================================================================================")
print("Model loaded successfully!")

explanation = """
Voltaire’s
original name was François-Marie Arouet.
He was French. He
opined that along
with objective truth
and chronology of
historical events
considering social
traditions, trade,
economy, agriculture,
etc. was also equally
important in historiography. It gave rise
to the thought that understanding all
aspects of human life is important for
history writing. Thus, it is said that
Voltaire was the founder of modern
historiography. """
#Question generation prompt template
template = f"""
    You are a tutor. Based on the following explanation, create ONE clear question
    that checks whether the student has understood the concept.

    Explanation: {explanation}
    Question:
    """

response = llm.invoke(template)

start_index = response.find("Question:")
if start_index != -1:
    final_response = response[start_index + len("Question:"):].strip()
else:
    final_response = response.strip()


print("========================================================================================================")
print(final_response)

student_answer = input("Student's Answer: ")

prompt = f""" You are a tutor.

    Explanation: {explanation}
    Question: {final_response}
    Student's Answer: {student_answer}

    Tasks:
    1. Check if the student’s answer is correct or not (be fair).
    2. Point out mistakes or missing parts if any.
    3. Give encouraging, constructive feedback.
    4. If wrong, provide the correct answer.

    Feedback:
    """

evaluation = llm.invoke(prompt)
print("========================================================================================================")
start = evaluation.find("Feedback:")
if start != -1:
    final_evaluation = evaluation[start + len("Feedback:"):].strip()
else:
    final_evaluation = evaluation.strip()

print(final_evaluation)