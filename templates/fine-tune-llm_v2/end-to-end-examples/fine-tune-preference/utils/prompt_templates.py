PROMPT_TEMPLATE_QUESTION_GENERATION = """Given the following text, generate five multiple choice questions with the following format. The questions must be simple enough to be answerable with only general important details that would be included in a short two sentence summary of the text. The questions must be only answerable when given the text and should not be answerable with common knowledge. Do not write questions about minute details in the text, only the most important points.

Format:
Q1) Question
A. Choice 1
B. Choice 2
C. Choice 3
D. Choice 4
E. Choice 5

Q1 Answer: A/B/C/D/E

Q2) Question
A. Choice 1
B. Choice 2
C. Choice 3
D. Choice 4
E. Choice 5

Q2 Answer: A/B/C/D/E

etc...

Text:
{text}"""

PROMPT_TEMPLATE_SUMMARY = """Given the following text, create a very short summary that is at most 2 sentences.

Text:
{text}"""

PROMPT_TEMPLATE_MCQ_ANSWERING = """You will be given a text passage followed by multiple choice questions about that passage. Your task is to answer these questions based solely on the information provided in the text. Do not use any external knowledge or make inferences beyond what is explicitly stated in the passage.

Here is the text:

{summary_generation_raw_model_output}

Here are the questions:

{qa_generation_questions}

Carefully read the text and each question. For each question:

1. Analyze whether the text contains the necessary information to answer the question.
2. If the information is present, select the correct answer from the given options.
3. If the information is not present or is insufficient to determine the answer, respond with "Unsure."

Format your answers as follows:

Q1) [Your answer (A./B./C./D./E.) or "Unsure."]
Q2) [Your answer (A./B./C./D./E.) or "Unsure."]
Q3) [Your answer (A./B./C./D./E.) or "Unsure."]
(Continue for all questions)

Remember:
- Only use information explicitly stated in the given text.
- Do not make inferences or use external knowledge.
- If the text does not provide enough information to answer a question confidently, respond with "Unsure."
- Provide only the letter of the correct answer (A, B, C, etc.) or "Unsure." for each question."""
