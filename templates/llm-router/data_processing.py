import re


def preprocess_nectar(df, model, model_name):
    # Filter to include only the first turn, good-natured responses, and responses that contain the specified model
    conditions = (df['turns'] == 1) & df['good_natured'] & df['answers'].apply(lambda ans: any(model == a.get('model') for a in ans))
    filtered_df = df[conditions]
    
    # Extract the answer for the specified model from the list of all answers
    filtered_df[model_name] = filtered_df['answers'].apply(lambda row: next((item['answer'] for item in row if item['model'] == model), None))
    
    # Clean user prompts
    pattern_start = re.compile(r"^\s+[Hh]uman:\s+")
    pattern_end = re.compile(r"\s+[Aa]ssistant:\s+$")
    
    filtered_df['prompt'] = filtered_df['prompt'].apply(lambda prompt: pattern_end.sub("", pattern_start.sub("", prompt)).strip())

    # Drop unnecessary columns
    filtered_df.drop(columns=['answers', 'num_responses', 'turns', 'good_natured'], inplace=True)

    return filtered_df

