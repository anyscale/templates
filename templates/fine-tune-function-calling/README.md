# Fine-tuning for Function calling on custom data.

**⏱️ Time to complete**: 5 hours for 7/8B models (11 hours for 13B, 27 hours for 70B)

Function calling is an important capability of large language models. Connecting your model to external tools is at the heart of many LLM applications. In Anyscale Endpoints, you can use the [function calling API](https://docs.anyscale.com/preview/endpoints/text-generation/function-calling) to enable get a quick access on this feature on a select number of models. This is made possible [through JSON mode](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features). However, it is beneficial to have *native* function calling capabilities in your model through fine-tuning on a relevant function calling dataset. JSON-mode-based function calling can only guarantee that the output is in the right schema, and can also be more expensive than a regular chat completion. However, fine-tuning on a function calling dataset can improve the model's capabilities with intent recognition (understanding when to call and when not to call a tool) and function call accuracy (employing the right function with accurate parameters) in addition to structured data formatting (formatting the function call json in the correct schema).  Fine-tuning would also be the only systematic way to improve performance on use-case-specific data. 

In this example, we demonstrate fine-tuning on [Glaive's function calling dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2?row=0) using Anyscale Endpoints. The goal for this example is to serve as a blue-print for performing data processing, training, and evaluation on open source LLMs for specific tasks like function calling, in the most effective way. The mentioned dataset consists of about 113,000 examples of synthetically generated function calling data. The dataset composition is given below:

<p align="center">
  <img src="./assets/distr_glaive_pie.png" alt="Distribution">
</p>


# Table of Contents
1. [Data Preprocessing](#step-1-data-preprocessing): In this section we will cover how we can use ray data to clean and format our raw dataset properly and create a train, valid, and test datasets.
2. [Finetuning](#step-2-fine-tuning): This section will cover a few different ways you can fine-tune LLMs via Anyscale.
3. [Serving](#step-3-serving): This section will cover how we can serve the fine-tuned model via Anyscale.
4. [Evaluation](#step-4-evaluation): The section will cover blue-print for evaluation and comparison to closed source models like OpenAI GPT-4 models.

First, let's make the necessary imports


```python
import datasets
import ray.data 
import openai
```


```python
from fc_utils.data_format import TOOL_CALL_TAGS, TOOL_RESULT_TAGS, TOOL_LIST_TAGS, DatasetFormat
from fc_utils.preprocessing import preprocess_to_openai_format, pprint_example, preprocess_to_anyscale_format, save_to_jsonl
from fc_utils.response_parsers import OpenAIResponseParser, AnyscaleResponseParser
from fc_utils.eval_utils import evaluate_model, Model
from fc_utils.test_utils import get_evaluation_dataset
from fc_utils.plot_utils import plot_results
```

# Step 1: Data Preprocessing
We'll use Ray Data for scalable data processing. First let's load the dataset from the HuggingFace Hub and inspect a few entries


```python
hf_ds = datasets.load_dataset("glaiveai/glaive-function-calling-v2", split="train").shuffle(seed=21) 
hf_ds_subset =  hf_ds.select(range(int(len(hf_ds)*0.10))) # sample only 10% of the dataset
ray_ds = ray.data.from_huggingface(hf_ds_subset)
first_ex = ray_ds.take(1)[0]
```

    2024-05-19 20:55:21,547	INFO worker.py:1740 -- Started a local Ray instance. View the dashboard at [1m[32m127.0.0.1:8265 [39m[22m
    2024-05-19 20:55:25,061	INFO dataset.py:2370 -- Tip: Use `take_batch()` instead of `take() / show()` to return records in pandas or numpy batch format.
    2024-05-19 20:55:25,063	INFO streaming_executor.py:112 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-05-19_20-55-19_472957_88751/logs/ray-data
    2024-05-19 20:55:25,064	INFO streaming_executor.py:113 -- Execution plan of Dataset: InputDataBuffer[Input] -> LimitOperator[limit=1]



    - limit=1 1:   0%|          | 0/1 [00:00<?, ?it/s]



    Running 0:   0%|          | 0/1 [00:00<?, ?it/s]



```python
pprint_example(first_ex, dataset_format=DatasetFormat.GLAIVE)
```

    [91msystem: [0mSYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
    {
        "name": "create_reminder",
        "description": "Create a reminder for a specific date and time",
        "parameters": {
            "type": "object",
            "properties": {
                "reminder_text": {
                    "type": "string",
                    "description": "The content of the reminder"
                },
                "reminder_date": {
                    "type": "string",
                    "format": "date",
                    "description": "The date of the reminder"
                },
                "reminder_time": {
                    "type": "string",
                    "format": "time",
                    "description": "The time of the reminder"
                }
            },
            "required": [
                "reminder_text",
                "reminder_date",
                "reminder_time"
            ]
        }
    }
    
    [92mchat: [0mUSER: I need to set a reminder for my doctor's appointment.
    
    
    ASSISTANT: Sure, I can help with that. Could you please provide me with the date and time of your appointment? <|endoftext|>
    
    
    USER: The appointment is on 2022-09-15 at 10:00 AM.
    
    
    ASSISTANT: <functioncall> {"name": "create_reminder", "arguments": '{"reminder_text": "Doctor's appointment", "reminder_date": "2022-09-15", "reminder_time": "10:00"}'} <|endoftext|>
    
    
    FUNCTION RESPONSE: {"status": "success", "message": "Reminder for 'Doctor's appointment' on 2022-09-15 at 10:00 AM has been created successfully."}
    
    
    ASSISTANT: Your reminder for the doctor's appointment on 2022-09-15 at 10:00 AM has been created successfully. You will be notified at the specified time. <|endoftext|>
    
    
    
    


If you notice, each sample has two entries: system and chat. This dataset is already formatted in specific way (e.g. using USER, \<|endoftext|\> and other tokens). To enable fine-tuning on various open source models we need to convert each row to a more general format like the OpenAI chat format, which is the preferred format for fine-tuning instruction-tuned models on Anyscale ([dataset format guide](https://docs.endpoints.anyscale.com/endpoints/fine-tuning/dataset-prep)). Let's first bring this dataset into the conversation format and inspect how that looks like:


```python
import fc_utils.preprocessing
import importlib
importlib.reload(fc_utils.preprocessing)
from fc_utils.preprocessing import pprint_example
```


```python
# initial preprocessing to get to the OpenAI format
openai_fmt_ds = preprocess_to_openai_format(ray_ds)
first_ex = openai_fmt_ds.take(1)[0] 
```

    2024-05-19 20:57:57,797	INFO streaming_executor.py:112 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-05-19_20-55-19_472957_88751/logs/ray-data
    2024-05-19 20:57:57,798	INFO streaming_executor.py:113 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[Map(glaive_to_openai)->Filter(<lambda>)->Filter(filter_func)] -> LimitOperator[limit=1]



    - Map(glaive_to_openai)->Filter(<lambda>)->Filter(filter_func) 1:   0%|          | 0/1 [00:00<?, ?it/s]



    - limit=1 2:   0%|          | 0/1 [00:00<?, ?it/s]



    Running 0:   0%|          | 0/1 [00:00<?, ?it/s]



```python
import fc_utils.preprocessing
import importlib
importlib.reload(fc_utils.preprocessing)
from fc_utils.preprocessing import pprint_example
```


```python
pprint_example(first_ex, dataset_format=DatasetFormat.OPENAI) # inspect one example
```

    [92mMessages: [0m
    	[91msystem: [0mYou are a helpful assistant.
    	[92muser: [0mI need to set a reminder for my doctor's appointment.
    	[94massistant: 
    		content: [0mSure, I can help with that. Could you please provide me with the date and time of your appointment? 
    		[94mtool_calls: [0m[]
    	[92muser: [0mThe appointment is on 2022-09-15 at 10:00 AM.
    	[94massistant: 
    		content: [0m
    		[94mtool_calls: [0m[{'function': {'arguments': '{"reminder_text": "Doctors appointment", "reminder_date": "2022-09-15", "reminder_time": "10:00"}', 'name': 'create_reminder'}, 'id': 'call_3d8bc9df5a3f47b9a836bdee', 'type': 'function'}]
    	[93mtool: [0m{"name": "create_reminder", "content": "{\"status\": \"success\", \"message\": \"Reminder for 'Doctor's appointment' on 2022-09-15 at 10:00 AM has been created successfully.\"}", "tool_call_id": "call_3d8bc9df5a3f47b9a836bdee"}
    	[94massistant: 
    		content: [0mYour reminder for the doctor's appointment on 2022-09-15 at 10:00 AM has been created successfully. You will be notified at the specified time. 
    		[94mtool_calls: [0m[]
    [95mTools: [0m[{"type": "function", "function": {"name": "create_reminder", "description": "Create a reminder for a specific date and time", "parameters": {"type": "object", "properties": {"reminder_text": {"type": "string", "description": "The content of the reminder"}, "reminder_date": {"type": "string", "format": "date", "description": "The date of the reminder"}, "reminder_time": {"type": "string", "format": "time", "description": "The time of the reminder"}}, "required": ["reminder_text", "reminder_date", "reminder_time"]}}}]
    


We'll now further process this conversation format and make it compatible with Anyscale Endpoints. We'll make sure of special indicators "\[TOOL_CALLS\]" and "\[/TOOL_CALLS\] to format assistant tool calls into the message "content" field. The role "tool" will be converted to the role "user" with a special indicator to highlight that this is a tool response. Further, the tool list will be included in the system prompt with special indicators. The following code block handles the necessary preprocessing.


```python
# complete preprocessing step
processed_ds = preprocess_to_anyscale_format(openai_fmt_ds)
first_ex = processed_ds.take(1)[0]
```

    2024-05-19 21:00:20,978	INFO streaming_executor.py:112 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-05-19_20-55-19_472957_88751/logs/ray-data
    2024-05-19 21:00:20,979	INFO streaming_executor.py:113 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[Map(glaive_to_openai)->Filter(<lambda>)->Filter(filter_func)->Map(openai_to_anyscale)->Filter(filter_func)] -> LimitOperator[limit=1]



    - Map(glaive_to_openai)->Filter(<lambda>)->Filter(filter_func)->Map(openai_to_anyscale)->Filter(filter_func) 1…



    - limit=1 2:   0%|          | 0/1 [00:00<?, ?it/s]



    Running 0:   0%|          | 0/1 [00:00<?, ?it/s]



```python
pprint_example(first_ex, dataset_format=DatasetFormat.ANYSCALE) # inspect one example
```

    [92mMessages: [0m
    	[91msystem: [0mYou are a helpful assistant.[TOOL_LIST] [{"type": "function", "function": {"name": "create_reminder", "description": "Create a reminder for a specific date and time", "parameters": {"type": "object", "properties": {"reminder_text": {"type": "string", "description": "The content of the reminder"}, "reminder_date": {"type": "string", "format": "date", "description": "The date of the reminder"}, "reminder_time": {"type": "string", "format": "time", "description": "The time of the reminder"}}, "required": ["reminder_text", "reminder_date", "reminder_time"]}}}] [/TOOL_LIST]
    	[92muser: [0mI need to set a reminder for my doctor's appointment.
    	[94massistant: [0mSure, I can help with that. Could you please provide me with the date and time of your appointment? 
    	[92muser: [0mThe appointment is on 2022-09-15 at 10:00 AM.
    	[94massistant: [0m[TOOL_CALLS] [{"function": {"arguments": "{\"reminder_text\": \"Doctors appointment\", \"reminder_date\": \"2022-09-15\", \"reminder_time\": \"10:00\"}", "name": "create_reminder"}, "id": "call_7d85fa44c8c54ede90954b4b", "type": "function"}] [/TOOL_CALLS]
    	[92muser: [0m[TOOL_RESULT] {"name": "create_reminder", "content": "{\"status\": \"success\", \"message\": \"Reminder for 'Doctor's appointment' on 2022-09-15 at 10:00 AM has been created successfully.\"}", "tool_call_id": "call_7d85fa44c8c54ede90954b4b"} [/TOOL_RESULT]
    	[94massistant: [0mYour reminder for the doctor's appointment on 2022-09-15 at 10:00 AM has been created successfully. You will be notified at the specified time. 
    


Let's make a train, validation and test split and save the datasets in the `jsonl` format.


```python
train_ds, val_ds, test_ds = processed_ds.split_proportionately([0.8, 0.1])
test_ds, _  = test_ds.split_at_indices([200]) # restrict to 200 examples for testing
```

    2024-05-19 21:00:38,320	INFO streaming_executor.py:112 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-05-19_20-55-19_472957_88751/logs/ray-data
    2024-05-19 21:00:38,320	INFO streaming_executor.py:113 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[Map(glaive_to_openai)->Filter(<lambda>)->Filter(filter_func)->Map(openai_to_anyscale)->Filter(filter_func)]



    - Map(glaive_to_openai)->Filter(<lambda>)->Filter(filter_func)->Map(openai_to_anyscale)->Filter(filter_func) 1…



    Running 0:   0%|          | 0/1 [00:00<?, ?it/s]



```python
# inspect final counts
train_ds.count(), val_ds.count(), test_ds.count()
```




    (9013, 1127, 200)




```python
# save the datasets to jsonl format
save_to_jsonl(train_ds, "glaiveai-function-calling-v2-train.jsonl")
save_to_jsonl(val_ds, "glaiveai-function-calling-v2-val.jsonl")
save_to_jsonl(test_ds, "glaiveai-function-calling-v2-test.jsonl")
```

# Step 2: Fine-tuning 

For fine-tuning, you have two options with Anyscale:
1. Fine-tuning on the Anyscale Platform through our fine-tuning template 
    - This would be the preferred route for those wishing to get more flexibility in choice of models and hyperparameters, better monitoring, etc.
2. Fine-tuning through Anyscale's serverless endpoints
    - A quick and easy way to fine-tune a model via an OpenAI compatiable SDK.

For this guide, we will use `Llama-3-8B-Instruct` as the base model for fine-tuning.



## Step 2(a): Fine-tuning on the Anyscale Platform

Head over to the Anyscale Platform: https://console.anyscale.com/v2 and spin up the "Fine-tune LLMs" template (under "AI application templates")

<p align="center">
  <img src="./assets/templates.png" alt="Templates">
</p>



Follow the instructions to run your fine-tuning job.

## Step 2(b): Fine-tuning through serverless endpoints
First, obtain your credentials from the [Anyscale platform](https://console.anyscale.com/v2/) and upload the training and validation files in the [fine-tuning tab](https://console.anyscale.com/v2/fine-tuning?fine-tuning-tab=files). Make a note of the file IDs for each. This will be passed to the fine-tuning job.


```python
ANYSCALE_API_KEY = "esecret_yourKeyHere"  # from https://console.anyscale.com/credentials
ANYSCALE_API_BASE = "https://api.endpoints.anyscale.com/v1"
```


```python
# Anyscale Endpoints are OpenAI compatible
client = openai.OpenAI(
    base_url = ANYSCALE_API_BASE,
    api_key = "esecret_yourKeyHere" 
)
```

Let's now launch a fine-tuning job for 4 epochs. The expected time for this job is < 3 hours.


```python
# other parameters like context length will be chosen appropriately based on dataset size
client.fine_tuning.jobs.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    hyperparameters={"n_epochs": 4},
    # replace with the actual file ids!
    training_file="file_trainingFileId",
    validation_file="file_validationFileId",
)
```

# Step 3: Serving

## Step 3(a): Finetuned on the Anyscale Platform

Make a note of the final checkpoint after fine-tuning (this should be the last line in the logs). You can now spin up the "Deploy LLMs" template which has all the instructions and required dependencies to serve your finetuned model efficiently. You will find the tutorials on [serving LoRA models](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/lora/DeployLora.ipynb) (if applicable) and on deploying a [custom model](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/CustomModels.ipynb) helpful. Once you have set up your fine-tuned model as an Anyscale Service, note down the base URL and API key and place them here.


```python
# to be run only if you finetuned on the Anyscale platform
ANYSCALE_API_KEY="service-api-key-here"
ANYSCALE_API_BASE="service-url-here" 
if not ANYSCALE_API_BASE.endswith("/"):
    ANYSCALE_API_BASE += "/"
ANYSCALE_API_BASE += "v1"
# enter the model id here. This would be different depending on whether you performed LoRA or full parameter fine-tuning.
# Example: meta-llama/Meta-Llama-3-8B-Instruct:mysuffix:myid 
MODEL_ID = "ModelIdHere"
```

## Step 3(b): Finetuned through serverless endpoints

To serve the fine-tuned model, you just need to navigate to the "Serving" section on the Anyscale Platform. Your fine-tuned model should already be visible in the list of available models! 

<p align="center">
  <img src="./assets/serving_endpoints.png" alt="Serve Endpoints">
</p>


As in the above image, click on the three dots and then click on "Query". This will provide you the starter code to interact with the model via curl, python, etc. 

<p align="center">
  <img src="./assets/serve_api_key.png" alt="API Key">
</p>



```python
## This is only if you finetuned through serverless endpoints
ANYSCALE_API_BASE = "https://api.endpoints.anyscale.com/v1"
ANYSCALE_API_KEY = "esecret_yourKeyHere"
MODEL_ID = "yourModelIdHere" # make sure to not add a stray slash "/"" at the end!
```

### (Optional) Try out the model via Playground

You can try out your new model in the Playground: https://console.anyscale.com/v2/playground . In the model dropdown, you should be able to see your finetuned model as shown below

<p align="center">
  <img src="./assets/playground.png" alt="Playground">
</p>

# Step 4: Evaluation

Let's evaluate our trained model with GPT-4 as a baseline. 


## Evaluation strategy

Evaluation of function calling capability is non-trivial, given that we're looking to extract structured data from an inherently unpredictable and unstructured stream of text. We will use the following simple evaluation strategy: The models are evaluated on the accuracy metric and their responses are graded as accurate if their response for each assistant entry in the conversation is correct. An assistant response is graded as correct under the below conditions:
1. In case the ground truth response contains no function call, then the model's response should also not have a function call. 
2. In case the ground truth response contains a function call, then the model's response should also have a function call. We do not check for the content of the response here. The assistant function call should further have the correct function name and the correct function arguments. 

The following psuedocode shows some of the different branching conditions considered during evaluation:

```
if(ground_truth has no function call):
    correct = (response has no function call)
else
    if response has no function call: 
        correct = False
    else
          if response.function_name != gt.function_name:
                correct = False
          else
                correct = (response.argument_dict == gt.argument_dict)
```



## Dataset formatting
  
We process our test dataset individually for our finetuned model on Anyscale and for GPT-4:
- For GPT-4, we undo some of the preprocessing previously done to get back the conversation in each example into the OpenAI format. All expected assistant responses in the dataset are processed to have the `"content"` and the `"tool_calls"` field. 
- We follow the same preprocessing as during training for the Anyscale hosted model. However, for the expected assistant response, we process it in the same way as GPT-4 (i.e parse all tool calls and store them in a separate `"tool_calls"` field).


```python
# preprocess the test dataset for evaluation
eval_ds_openai = get_evaluation_dataset(test_ds, TOOL_CALL_TAGS, TOOL_RESULT_TAGS, TOOL_LIST_TAGS, DatasetFormat.OPENAI)
eval_ds_anyscale = get_evaluation_dataset(test_ds, TOOL_CALL_TAGS, TOOL_RESULT_TAGS, TOOL_LIST_TAGS, DatasetFormat.ANYSCALE)
```


```python
pprint_example(eval_ds_anyscale[20], dataset_format=DatasetFormat.OPENAI)
```

    [92mMessages: [0m
    	[91msystem: [0mYou are a helpful assistant.[TOOL_LIST] [{"type": "function", "function": {"name": "generate_qr_code", "description": "Generate a QR code for a given text", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "The text to encode in the QR code"}}, "required": ["text"]}}}, {"type": "function", "function": {"name": "get_movie_details", "description": "Get the details of a movie", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "The title of the movie"}, "year": {"type": "integer", "description": "The release year of the movie"}}, "required": ["title"]}}}] [/TOOL_LIST]
    	[92muser: [0mHi, I need a QR code for my website. Can you help me with that?
    	[94massistant: 
    		content: [0mOf course, I can help you with that. Could you please provide me with the URL of your website? 
    		[94mtool_calls: [0mNone
    	[92muser: [0mSure, it's www.mywebsite.com.
    	[94massistant: 
    		content: [0mNone
    		[94mtool_calls: [0m[{'function': {'arguments': {'text': 'www.mywebsite.com'}, 'name': 'generate_qr_code'}, 'id': 'call_646f0233411b4796a25bf5b8', 'type': 'function'}]
    	[92muser: [0m[TOOL_RESULT] {"name": "generate_qr_code", "content": "{\"qr_code\": \"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAACXBIWXMAAAsTAAALEwEAmpwYAAAKT2lDQ1BQaG90b3Nob3AgSUN... (truncated)\"}", "tool_call_id": "call_646f0233411b4796a25bf5b8"} [/TOOL_RESULT]
    	[94massistant: 
    		content: [0mI have generated the QR code for your website. You can download it using this link: [QR Code](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAACXBIWXMAAAsTAAALEwEAmpwYAAAKT2lDQ1BQaG90b3Nob3AgSUN... (truncated)) 
    		[94mtool_calls: [0mNone
    


## Evaluate

For evaluation, we initialise two parsers - one for each model - to handle obtaining chat completions from the respective API and parsing the result. Then, our evaluation logic takes care of matching the assistant response with the expected response and, if the response is incorrect, making note of the type of error (wrong intent, wrong function name, etc). Populate the API keys below and run the below code blocks to get evaluation results:


```python
# enter your openai key below.
OPENAI_API_KEY = "yourApiKeyHere" 
# enter your Anyscale key below. If you finetuned through Anyscale endpoints, you can get the key here: https://console.anyscale.com/credentials. Otherwise, you should use the key from your Anyscale Service
ANYSCALE_API_KEY = "yourApiKeyHere" 
```


```python
# initialize parsers
openai_parser = OpenAIResponseParser(api_key=OPENAI_API_KEY, api_base="https://api.openai.com/v1", model="gpt-4", tool_call_tags=TOOL_CALL_TAGS)
anyscale_parser = AnyscaleResponseParser(api_key=ANYSCALE_API_KEY, api_base=ANYSCALE_API_BASE, model=MODEL_ID, tool_call_tags=TOOL_CALL_TAGS) 
```


```python
# evaluate both models and plot the results
results_gpt, accuracy_gpt = evaluate_model(eval_ds_openai, openai_parser, Model.GPT)
results_finetuned, accuracy_finetuned = evaluate_model(eval_ds_anyscale, anyscale_parser, Model.FINETUNED)
print("GPT-4 Accuracy: ", accuracy_gpt)
print("Fine-tuned Model Accuracy: ", accuracy_finetuned)
plot_results(results_finetuned, results_gpt)
```

Your final fine-tuned model should be able to rival GPT-4 level performance on this dataset. In fact, performance can be higher, due to the fact that the test dataset construction was straightforward. Here's how your error analysis plot might look like for `Llama-3-8B-Instruct`:

<p align="center">
  <img src="./assets/error_analysis.png?version=1" alt="Error Analysis">
</p>

# Summary

Congrats! You have now fine-tuned an open source model that can rival GPT-4 on function calling. As a quick recap, here's what we demonstrated in this notebook:
1. Preprocesing a function calling dataset into a conversational format
2. Fine-tuning a language model through either the Anyscale Platform or through Anyscale Endpoints
3. Serving the fine-tuned model on Anyscale
4. Evaluating the model against GPT-4 and analysing the results.
