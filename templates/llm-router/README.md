```python
%load_ext autoreload
%autoreload 2
```

# Introduction

When building Large Language Model (LLM) applications, we strive to balance between achieving the highest response quality while adhering to a limited cost budget. Closed models like GPT-4 are renowned for their superior quality, but they can become prohibitively expensive, especially when handling a large volume of queries. On the other hand, Open Source Software (OSS) models are more cost-effective but may not deliver the same quality, particularly for complex or domain-specific queries.

A "smart router" addresses this challenge by processing user queries and deciding whether to route them to a closed LLM or an OSS LLM, depending on the query's complexity or domain. Here’s a schematic representation of a smart router:

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/llm-router/assets/llm-router-flowchart_2.png" alt="Smart Router" width="800"/>
</div>

Given a set of user queries, a smart router enables generating high-quality LLM responses while minimizing the overall cost.

# Approach

In this tutorial, we'll demonstrate how to train a smart router on the Anyscale platform. We make the following design choices:

1. **Model Choices**: We’ll use GPT-4 as an example of a closed LLM and Mixtral-8x7B as the OSS LLM, so our smart router will route between these two models.
2. **Response Quality Rating**: We'll quantify the quality of an LLM response on a scale of 1 to 5 stars, with higher scores indicating better quality. For simplicity, we'll assume that GPT-4 always achieves a 5-star rating, so it serves as a reference for Mixtral-8x7B.
3. **Smart Router Model**: We'll finetune a Llama3-8B model as our smart router and leverage Anyscale's powerful API. Our research (see our [arXiv paper](put link to arxiv paper)) shows that this model offers superior routing performance compared to smaller architectures.

More concretely, the objective of a smart router is to direct simple queries to Mixtral-8x7B, thereby maintaining high overall response quality (e.g., an average score of 4.8/5) while significantly reducing costs (e.g., by 50%).





# Table of Contents

1. [**Prepare Labeled Data**](#generate-labeled-data): We describe how to generate synthetic labeled data to train the smart router model.

2. [**Finetune a Router Model**](#finetune-router-model): We show how to train a smart router by finetuning an LLM classifier using Anyscale's finetuning API.

3. [**Offline Evaluation**](#offline-eval): We load the model from the training checkpoint, run batch inference, and evaluate its performance.

**Time to complete**: 40 minutes


# Step 1: Prepare Labeled Data <a id="generate-labeled-data"></a>

Our smart router essentially functions as a binary classifier, deciding whether to route a query to GPT-4 or Mixtral-8x7B based on the query text. Initially, we considered labeled data in the format `(query, routing_label)`, where `routing_label` is 1 if the query should be routed to Mixtral-8x7B and 0 if it should be routed to GPT-4.

However, our early experiments revealed that *binary labels do not provide sufficient signal for training a robust router model*. Therefore, we adopted a different labeling approach using a *1-5 scoring system*, which reflects how well Mixtral-8x7B can effectively respond to the user's query. More specifically:

- **4-5**: Mixtral-8x7B produces a very strong answer, showing deep understanding, creativity, detailed insight, and high relevance.
- **3**: Mixtral-8x7B provides an adequate answer with moderate detail, relevance, and factual accuracy.
- **1-2**: Mixtral-8x7B struggles to produce a strong answer due to the question's difficulty, vagueness, or the model's limitations.

We use labeled samples in the format `(query, score_label)`. The `routing_label` can be derived from the `score_label` by setting a score threshold for quality, i.e. `routing_label = 1 if score_label >= 4 else 0`.

In the following, we will explain how we prepare our labeled dataset in detail.


## 1.1: Query Dataset

We want our smart router to be effective in open-ended chat domains. So, our first step is to collect a set of generic queries from the [Nectar dataset](https://huggingface.co/datasets/berkeley-nest/Nectar). We chose the Nectar dataset for two reasons: it combines queries from many different domains, including open-ended chat, and it has responses from many models, including over 191K responses from GPT-4.


```python
from src.utils import load_and_display_nectar

nectar_df = load_and_display_nectar()
```


    Downloading readme:   0%|          | 0.00/8.65k [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/517M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/182954 [00:00<?, ? examples/s]



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>answers</th>
      <th>turns</th>
      <th>num_responses</th>
      <th>source</th>
      <th>good_natured</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>\n\nHuman: 0.002 = 1000 \n1 = x?\n\nAssistant:</td>
      <td>[{'answer': 'To find the value of x, we can set up a proportion using the given information:

0.002/1000 = 1/x

To solve for x, we can cross multiply:

0.002 * x = 1000 * 1

0.002x = 1000

Dividing both sides by 0.002:

x = 1000 / 0.002

x = 500,000

Therefore, 1 is equal to 500,000 in this proportion.', 'model': 'gpt-3.5-turbo', 'rank': 1.0}, {'answer': 'To solve for x in this equation, you can use cross-multiplication. Here's how:

0.002 / 1 = 1000 / x

Cross-multiplication gives:

0.002 * x = 1000

To solve for x, divide both sides by 0.002:

x = 1000 / 0.002

So, x equals 500,000.', 'model': 'gpt-4-0613', 'rank': 2.0}, {'answer': 'This looks like a proportion. To solve for x, you can set up a proportion equation:

0.002 / 1 = 1000 / x

Now, cross-multiply:

0.002 \* x = 1 \* 1000

Then, solve for x:

0.002x = 1000

x = 1000 / 0.002

x = 500,000', 'model': 'gpt-3.5-turbo', 'rank': 3.0}, {'answer': 'If 0.002 equals 1000, then 1 would equal 500,000.', 'model': 'gpt-4', 'rank': 4.0}, {'answer': 'I apologize, but the equation you provided is not correct.

0.002 is equal to 0.002, not 1000.

1 is equal to x, not a number.

Can you please provide a correct equation or problem for me to solve?', 'model': 'llama-2-7b-chat', 'rank': 5.0}, {'answer': '0.001 = x', 'model': 'gpt-3.5-turbo-instruct', 'rank': 6.0}, {'answer': 'It seems like you are asking for the value of x in the equation x = 1/0.002. 

To solve this equation, you can divide both sides by 0.002 to get: 

x = 1/0.002 
x = 500 

Therefore, x = 500.', 'model': 'mistral-7b-instruct-v0.1', 'rank': 7.0}]</td>
      <td>1</td>
      <td>7</td>
      <td>[sharegpt]</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    Number of queries with GPT-4 responses: 191487


## 1.2 Data Preprocessing

We will use a subset of the Nectar data that includes responses from GPT-4, as these will be used to generate scores (as seen below). We will process this data by focusing on single-turn conversations, filtering for good-natured interactions, and cleaning up the prompts and responses to maintain high quality. Additionally, we will sample a small subset from the dataset for the purpose of this tutorial; however, you can skip sampling to work with the full dataset.


```python
from src.utils import preprocess_nectar

nectar_gpt4_df = preprocess_nectar(
    nectar_df, model="gpt-4", response_column="gpt4_response"
)

# Sample a small subset from the dataset for the purpose of this tutorial
N_SUBSET = 30
dataset_df = nectar_gpt4_df.sample(N_SUBSET, random_state=42)
```

### Dataset overview with GPT-4 responses


```python
display(dataset_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>source</th>
      <th>gpt4_response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6062</th>
      <td>Based on the features mentioned, which hotel d...</td>
      <td>[evol_instruct]</td>
      <td>Based on the features mentioned, Hotel A seems...</td>
    </tr>
    <tr>
      <th>113830</th>
      <td>Provide step-by-step instructions on how to cr...</td>
      <td>[ultrachat]</td>
      <td>Sure, here's a simple step-by-step guide on ho...</td>
    </tr>
    <tr>
      <th>138869</th>
      <td>What are the 10 largest cities in the US by po...</td>
      <td>[lmsys-chat-1m]</td>
      <td>As of the most recent data available, the 10 l...</td>
    </tr>
    <tr>
      <th>169249</th>
      <td>Write a comparison essay of at least 500 words...</td>
      <td>[ultrachat]</td>
      <td>Title: A Comparative Analysis of Driving a Car...</td>
    </tr>
    <tr>
      <th>116934</th>
      <td>Q: You are provided with an "Event", "Intent" ...</td>
      <td>[flan_v2_niv2]</td>
      <td>PersonX might feel satisfied or content using ...</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3 Data Labeling

We don't have human labels for scores, so we will use the [LLM-as-a-Judge approach](https://arxiv.org/abs/2306.05685). GPT-4 will act as an evaluator, reviewing the query and Mixtral's response to provide a score from 1-5. As shown in the paper, the most robust way to get labels is by providing a reference answer for comparison. Here, GPT-4's own response serves as the reference, and Mixtral's response is evaluated against it.

There are two main steps in this process:
1. **Generate Mixtral-8x7B responses for all queries**: We will use an online batch-inference method utilizing Ray and Anyscale endpoints.
2. **Generate LLM-as-a-Judge labels**: We will ask GPT-4 to evaluate the Mixtral responses against its own reference answers and provide a score from 1-5.

### Generate Mixtral-8x7B Responses


```python
import yaml
from src.online_inference import generate_mixtral_responses

# store ANYSCALE_API_KEY: "your_api_key_here" in /home/ray/keys.yaml
with open("/home/ray/keys.yaml") as config_file:
    keys = yaml.safe_load(config_file)
    anyscale_api_key = keys["ANYSCALE_API_KEY"]

dataset_df = generate_mixtral_responses(
    dataset_df, anyscale_api_key, response_column="mixtral_response"
)
```

    Starting batch inference on 30 queries...


    2024-06-24 17:11:56,396	INFO worker.py:1568 -- Connecting to existing Ray cluster at address: 10.0.0.30:6379...
    2024-06-24 17:11:56,404	INFO worker.py:1744 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-3mbbhc76us4jta3ixstn8kjsxy.i.anyscaleuserdata.com [39m[22m
    2024-06-24 17:11:56,412	INFO packaging.py:358 -- Pushing file package 'gcs://_ray_pkg_e6620ba04eb087533e81bed1366800219d71d38a.zip' (0.49MiB) to Ray cluster...
    2024-06-24 17:11:56,414	INFO packaging.py:371 -- Successfully pushed file package 'gcs://_ray_pkg_e6620ba04eb087533e81bed1366800219d71d38a.zip'.


    # queries un-processed: 29, in-progress: 1, ready: 0
    # queries un-processed: 28, in-progress: 2, ready: 0
    # queries un-processed: 27, in-progress: 3, ready: 0
    # queries un-processed: 26, in-progress: 4, ready: 0
    # queries un-processed: 25, in-progress: 5, ready: 0
    # queries un-processed: 24, in-progress: 6, ready: 0
    # queries un-processed: 23, in-progress: 7, ready: 0
    # queries un-processed: 22, in-progress: 8, ready: 0
    # queries un-processed: 21, in-progress: 9, ready: 0
    # queries un-processed: 20, in-progress: 10, ready: 0
    # queries un-processed: 19, in-progress: 11, ready: 0
    # queries un-processed: 18, in-progress: 12, ready: 0
    # queries un-processed: 17, in-progress: 13, ready: 0
    # queries un-processed: 16, in-progress: 14, ready: 0
    # queries un-processed: 15, in-progress: 15, ready: 0
    # queries un-processed: 14, in-progress: 16, ready: 0
    # queries un-processed: 13, in-progress: 17, ready: 0
    # queries un-processed: 12, in-progress: 18, ready: 0
    # queries un-processed: 11, in-progress: 19, ready: 0
    # queries un-processed: 10, in-progress: 19, ready: 1
    # queries un-processed: 9, in-progress: 19, ready: 1
    # queries un-processed: 8, in-progress: 19, ready: 1
    # queries un-processed: 7, in-progress: 20, ready: 0
    # queries un-processed: 6, in-progress: 21, ready: 0
    # queries un-processed: 5, in-progress: 21, ready: 1
    # queries un-processed: 4, in-progress: 21, ready: 1
    # queries un-processed: 3, in-progress: 21, ready: 1
    # queries un-processed: 2, in-progress: 22, ready: 0
    # queries un-processed: 1, in-progress: 22, ready: 1
    # queries un-processed: 0, in-progress: 23, ready: 0
    # queries un-processed: 0, in-progress: 22, ready: 1
    # queries un-processed: 0, in-progress: 21, ready: 1
    # queries un-processed: 0, in-progress: 20, ready: 1
    # queries un-processed: 0, in-progress: 19, ready: 1
    # queries un-processed: 0, in-progress: 19, ready: 0
    # queries un-processed: 0, in-progress: 18, ready: 1
    # queries un-processed: 0, in-progress: 17, ready: 1
    # queries un-processed: 0, in-progress: 16, ready: 1
    # queries un-processed: 0, in-progress: 16, ready: 0
    # queries un-processed: 0, in-progress: 15, ready: 1
    # queries un-processed: 0, in-progress: 14, ready: 1
    # queries un-processed: 0, in-progress: 14, ready: 0
    # queries un-processed: 0, in-progress: 13, ready: 1
    # queries un-processed: 0, in-progress: 12, ready: 1
    # queries un-processed: 0, in-progress: 11, ready: 1
    # queries un-processed: 0, in-progress: 11, ready: 0
    # queries un-processed: 0, in-progress: 11, ready: 0
    # queries un-processed: 0, in-progress: 10, ready: 1
    # queries un-processed: 0, in-progress: 9, ready: 1
    # queries un-processed: 0, in-progress: 8, ready: 1
    # queries un-processed: 0, in-progress: 7, ready: 1
    # queries un-processed: 0, in-progress: 6, ready: 1
    # queries un-processed: 0, in-progress: 6, ready: 0
    # queries un-processed: 0, in-progress: 5, ready: 1
    # queries un-processed: 0, in-progress: 5, ready: 0
    # queries un-processed: 0, in-progress: 4, ready: 1
    # queries un-processed: 0, in-progress: 3, ready: 1
    # queries un-processed: 0, in-progress: 2, ready: 1
    # queries un-processed: 0, in-progress: 2, ready: 0
    # queries un-processed: 0, in-progress: 1, ready: 1
    # queries un-processed: 0, in-progress: 1, ready: 0
    # queries un-processed: 0, in-progress: 0, ready: 1
    Done in 21.97sec.


    [36m(autoscaler +26m40s)[0m Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.


### Dataset overview with Mixtral responses



```python
display(dataset_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>source</th>
      <th>gpt4_response</th>
      <th>mixtral_response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6062</th>
      <td>Based on the features mentioned, which hotel d...</td>
      <td>[evol_instruct]</td>
      <td>Based on the features mentioned, Hotel A seems...</td>
      <td>Based on the information provided, I would ne...</td>
    </tr>
    <tr>
      <th>113830</th>
      <td>Provide step-by-step instructions on how to cr...</td>
      <td>[ultrachat]</td>
      <td>Sure, here's a simple step-by-step guide on ho...</td>
      <td>Sure, I'd be happy to help you make a homemad...</td>
    </tr>
    <tr>
      <th>138869</th>
      <td>What are the 10 largest cities in the US by po...</td>
      <td>[lmsys-chat-1m]</td>
      <td>As of the most recent data available, the 10 l...</td>
      <td>I'm here to help! According to the most recen...</td>
    </tr>
    <tr>
      <th>169249</th>
      <td>Write a comparison essay of at least 500 words...</td>
      <td>[ultrachat]</td>
      <td>Title: A Comparative Analysis of Driving a Car...</td>
      <td>Title: The Great Debate: Driving a Car vs. Ri...</td>
    </tr>
    <tr>
      <th>116934</th>
      <td>Q: You are provided with an "Event", "Intent" ...</td>
      <td>[flan_v2_niv2]</td>
      <td>PersonX might feel satisfied or content using ...</td>
      <td>Person X probably feels comfortable and focus...</td>
    </tr>
  </tbody>
</table>
</div>


### Generate GPT-4-as-a-judge scores 

Let's first take a look at an example query we will send to GPT-4 for judgement


```python
from src.utils import inspect_llm_judge_queries

inspect_llm_judge_queries(dataset_df)
```

    [Instruction]
    Evaluate the AI assistant's proficiency in answering the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, adherence to real-world facts, depth, creativity, and level of detail of the response. You will be given a reference answer which is considered of high quality. Your assessment will have two lines: First line has a rating on a scale of 1 to 5 with a higher rating representing higher response quality. Follow strictly this format: "[[rating]]", for example: "[[3]]". Second line contains a short explanation of your rating.
    
    [Question]
    Q: You are provided with an "Event", "Intent" related to PersonX. Guess a reaction/reaction of PersonX about the given event and their intention.
    Event:PersonX uses ___ in class. Intent: 1) to use his prefered writing implement
    A:
    
    [Reference Answer]
    PersonX might feel satisfied or content using their preferred writing implement in class, as it aligns with their intention to utilize a comfortable and desired tool for writing. 
    Confidence: 85%
    
    [Assistant Answer]
     Person X probably feels comfortable and focused in class, as they are using their preferred writing implement. They might appreciate being able to use a tool that helps them express their thoughts and ideas more effectively.
    
    Guidelines for Rating:
     - High Rating (4-5): Reserved for responses that are very close to the quality of the reference or even better.
     - Medium Rating (3): Reserved for responses that have moderate quality compared to the reference.
     - Low Rating (1-2): Allocated to response that are much lower quality compared to the reference or completely wrong.
    
    Assessment:
    


Now, we apply a similar online batch-inference method to generate our labels.


```python
import yaml
from src.online_inference import generate_llm_judge_labels

# store OPENAI_API_KEY: "your_api_key_here" in /home/ray/keys.yaml
with open("/home/ray/keys.yaml") as config_file:
    keys = yaml.safe_load(config_file)
    openai_api_key = keys["OPENAI_API_KEY"]

dataset_df = generate_llm_judge_labels(dataset_df, openai_api_key)
```

    Starting batch inference on 30 queries...
    # queries un-processed: 29, in-progress: 1, ready: 0
    # queries un-processed: 28, in-progress: 2, ready: 0
    # queries un-processed: 27, in-progress: 3, ready: 0
    # queries un-processed: 26, in-progress: 4, ready: 0
    # queries un-processed: 25, in-progress: 5, ready: 0
    # queries un-processed: 24, in-progress: 5, ready: 1
    # queries un-processed: 23, in-progress: 5, ready: 1
    # queries un-processed: 22, in-progress: 6, ready: 0
    # queries un-processed: 21, in-progress: 6, ready: 1
    # queries un-processed: 20, in-progress: 7, ready: 0
    # queries un-processed: 19, in-progress: 7, ready: 1
    # queries un-processed: 18, in-progress: 8, ready: 0
    # queries un-processed: 17, in-progress: 9, ready: 0
    # queries un-processed: 16, in-progress: 9, ready: 1
    # queries un-processed: 15, in-progress: 9, ready: 1
    # queries un-processed: 14, in-progress: 9, ready: 1
    # queries un-processed: 13, in-progress: 9, ready: 1
    # queries un-processed: 12, in-progress: 9, ready: 1
    # queries un-processed: 11, in-progress: 9, ready: 1
    # queries un-processed: 10, in-progress: 9, ready: 1
    # queries un-processed: 9, in-progress: 9, ready: 1
    # queries un-processed: 8, in-progress: 9, ready: 1
    # queries un-processed: 7, in-progress: 10, ready: 0
    # queries un-processed: 7, in-progress: 9, ready: 1
    # queries un-processed: 6, in-progress: 9, ready: 1
    # queries un-processed: 5, in-progress: 10, ready: 0
    # queries un-processed: 5, in-progress: 10, ready: 0
    # queries un-processed: 5, in-progress: 9, ready: 1
    # queries un-processed: 4, in-progress: 9, ready: 1
    # queries un-processed: 3, in-progress: 9, ready: 1
    # queries un-processed: 2, in-progress: 9, ready: 1
    # queries un-processed: 1, in-progress: 9, ready: 1
    # queries un-processed: 0, in-progress: 9, ready: 1
    # queries un-processed: 0, in-progress: 8, ready: 1
    # queries un-processed: 0, in-progress: 8, ready: 0
    # queries un-processed: 0, in-progress: 8, ready: 0
    # queries un-processed: 0, in-progress: 7, ready: 1
    # queries un-processed: 0, in-progress: 7, ready: 0
    # queries un-processed: 0, in-progress: 6, ready: 1
    # queries un-processed: 0, in-progress: 5, ready: 1
    # queries un-processed: 0, in-progress: 4, ready: 1
    # queries un-processed: 0, in-progress: 4, ready: 0
    # queries un-processed: 0, in-progress: 3, ready: 1
    # queries un-processed: 0, in-progress: 2, ready: 1
    # queries un-processed: 0, in-progress: 2, ready: 0
    # queries un-processed: 0, in-progress: 1, ready: 1
    # queries un-processed: 0, in-progress: 0, ready: 1
    Done in 14.12sec.


### Dataset overview with score labels



```python
display(dataset_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>source</th>
      <th>gpt4_response</th>
      <th>mixtral_response</th>
      <th>mixtral_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6062</th>
      <td>Based on the features mentioned, which hotel d...</td>
      <td>[evol_instruct]</td>
      <td>Based on the features mentioned, Hotel A seems...</td>
      <td>Based on the information provided, I would ne...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>113830</th>
      <td>Provide step-by-step instructions on how to cr...</td>
      <td>[ultrachat]</td>
      <td>Sure, here's a simple step-by-step guide on ho...</td>
      <td>Sure, I'd be happy to help you make a homemad...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>138869</th>
      <td>What are the 10 largest cities in the US by po...</td>
      <td>[lmsys-chat-1m]</td>
      <td>As of the most recent data available, the 10 l...</td>
      <td>I'm here to help! According to the most recen...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>169249</th>
      <td>Write a comparison essay of at least 500 words...</td>
      <td>[ultrachat]</td>
      <td>Title: A Comparative Analysis of Driving a Car...</td>
      <td>Title: The Great Debate: Driving a Car vs. Ri...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>116934</th>
      <td>Q: You are provided with an "Event", "Intent" ...</td>
      <td>[flan_v2_niv2]</td>
      <td>PersonX might feel satisfied or content using ...</td>
      <td>Person X probably feels comfortable and focus...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


### Full Dataset
We have previously generated the full labeled datasets, created a train and validation splits, and published them as a public huggingface dataset `outellm/gpt4_dataset`. Let's load the dataset and explore the score distribution.



```python
from datasets import load_dataset
from src.utils import visualize_label_distribution

full_dataset_df = load_dataset("routellm/gpt4_dataset")
train_df = full_dataset_df["train"].to_pandas()
valid_df = full_dataset_df["validation"].to_pandas()

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(valid_df)}")
display(train_df.head())
visualize_label_distribution(train_df, key="mixtral_score")
```


    Downloading readme:   0%|          | 0.00/31.0 [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/290M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/26.7M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/109101 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/10000 [00:00<?, ? examples/s]


    Train size: 109101
    Validation size: 10000



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>source</th>
      <th>gpt4_response</th>
      <th>mixtral_response</th>
      <th>mixtral_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I'll give you a review, can you extract the fo...</td>
      <td>[lmsys-chat-1m]</td>
      <td>Sure, here's the analysis of the review:\n\n1....</td>
      <td>Food aspects and opinion words:\n\n1. Made to ...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Answer the following question: Claim: "Joker m...</td>
      <td>[flan_v2_cot]</td>
      <td>The answer is no.\nChain of thoughts: Stan Lee...</td>
      <td>The answer is no.\n\nChain of thoughts: While ...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TASK DEFINITION: In this task you will be give...</td>
      <td>[flan_v2_niv2]</td>
      <td>ZdoublexpropheciesS</td>
      <td>ZdoublexpropheciesS\n\nIn this task, you are a...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Detailed Instructions: In this task, you need ...</td>
      <td>[flan_v2_niv2]</td>
      <td>Yes</td>
      <td>No, 'station' is not the longest word in the s...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A guy pick up a woman Then he puts her down Ex...</td>
      <td>[sharegpt]</td>
      <td>This phrase could be interpreted as a joke bec...</td>
      <td>This joke is a play on words and relies on the...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_23_7.png)
    


Higher counts for 4-5 scores indicate that Mixtral-8x7B consistently produces high-quality responses, demonstrating its competitive performance compared to the June 2023 version of GPT-4, whose responses are logged in the Nectar dataset.

Let us assume that if the score is >= 4, we will route to the OSS model (indicating the response quality is good enough); otherwise, we will route to the closed model. Under this assumption, the data distribution looks like this:



```python
train_df["routing_label"] = train_df["mixtral_score"].apply(
    lambda x: 1 if x >= 4 else 0
)
visualize_label_distribution(train_df, key="routing_label")
```


    
![png](README_files/README_25_0.png)
    


# Step 2: Finetune a router model <a id="finetune-router-model"></a>

In this section, we will explain how to finetune an LLM as a smart router. While our data contains `gpt4_response` and `mixtral_response`, we will only use the pair (`query`, `mixtral_score`) for training. The goal is for the smart router to rely solely on the query text to determine which model to route to. Our approach is straightforward: we train a 5-way classifier to predict the `mixtral_score` from the `query`. At inference time, we will route to Mixtral if our router predicts a high score (i.e., 4-5) and to GPT-4 otherwise.


## 2.1 Data Preparation
We will discuss a few preprocessing steps to prepare the data for finetuning an LLM to be a smart router.

### Task Instructions
We use the instruction-following framework to finetune an LLM as a smart router. The task instructions guide the model to predict the score label for a given query. They ensure the model understands the evaluation criteria and can accurately assess the query's complexity and expected response quality.


```python
from src.utils import inspect_instructions

inspect_instructions(train_df)
```

    [Instruction]
    Based on the question provided below, predict the score an expert evaluator would give to an AI assistant's response, considering its helpfulness, relevance, adherence to facts, depth, creativity, and detail. Your prediction should infer the level of proficiency needed to address the question effectively. Use a scale from 1 to 5, where a higher score indicates a higher anticipated quality of response. Provide your prediction as: "[[predicted rating]]".
    
    Score criteria:
    - **4-5**: The AI assistant can produce a very strong answer, showing deep understanding, creativity, detailed insight, and high relevance.
    - **3**: The AI assistant can provide an adequate answer with moderate detail, relevance, and factual accuracy.
    - **1-2**: The AI assistant will struggle to produce a strong answer due to the question's difficulty, vagueness, or the assistant's limitations.
    
    [Question]
    {question}
    
    Prediction:
    


### API Data Format

To finetune the model, we must format the data to be compatible with [Anyscale's finetuning API](https://docs.anyscale.com/endpoints/fine-tuning/dataset-prep).



```python
from src.utils import prepare_ft_messages

train_df["messages"] = prepare_ft_messages(train_df, "mixtral_score")

# here's what the API data format looks like:
display(train_df["messages"].iloc[0])
```


    [{'role': 'system',
      'content': '[Instruction]\nBased on the question provided below, predict the score an expert evaluator would give to an AI assistant\'s response, considering its helpfulness, relevance, adherence to facts, depth, creativity, and detail. Your prediction should infer the level of proficiency needed to address the question effectively. Use a scale from 1 to 5, where a higher score indicates a higher anticipated quality of response. Provide your prediction as: "[[predicted rating]]".\n\nScore criteria:\n- **4-5**: The AI assistant can produce a very strong answer, showing deep understanding, creativity, detailed insight, and high relevance.\n- **3**: The AI assistant can provide an adequate answer with moderate detail, relevance, and factual accuracy.\n- **1-2**: The AI assistant will struggle to produce a strong answer due to the question\'s difficulty, vagueness, or the assistant\'s limitations.\n'},
     {'role': 'user',
      'content': "[Question]\nI'll give you a review, can you extract the food aspects and the opinion words of these aspects and analyze the sentiment of these opinion from this review? the review is:They tore the old NAME_1 down then built another one...? Anyway, they sell wine and beer and snacks and have a seating area inside and outside to eat. Besides gas, the big draw is the Made to Order food. I ordered some tacos and French toast sticks both were pretty good. I think I'd like to try more snacks.And they're open 24/7.\n\nPrediction:\n"},
     {'role': 'assistant', 'content': '[[4]]'}]


### Label Rebalancing

For classification tasks, it's recommended to train on label-balanced datasets to ensure models are not biased to a specific label. We will balance the dataset based on `routing_label`, as this is the label of primary interest.



```python
from src.utils import balance_dataset

balanced_train_df = balance_dataset(train_df, key="routing_label")
print(f"Train size: {len(balanced_train_df)}")
```

    Train size: 29504


### Subsample and Store Data

To expedite the time to run this tutorial, we will subsample 1,000 examples for training. We'll store the data in JSONL format to prepare for launching the finetuning job in the next section.


```python
n_sample = 1000
output_file = "/mnt/user_storage/train_data_sample.jsonl"

subsampled_df = balanced_train_df.sample(n=n_sample, random_state=42)
subsampled_df.to_json(output_file, orient="records", lines=True)
```

## 2.2 Fine-tune with Anyscale API

We will run a fine-tuning job using Anyscale's LLM finetuning API as an isolated job, similar to this [tutorial](https://github.com/anyscale/e2e-llm-workflows?tab=readme-ov-file#fine-tuning-1).

For this tutorial, we will perform full-parameter finetuning of Llama3-8B on the same 1,000 samples we showed earlier to debug the training dynamics and ensure the model can fit the training set. Below, we present the training and job configurations before submitting the training job.



```python
# View the full-param finetuning configuration for llama-3-8B
!cat configs/ft_config_debug.yaml
```

    model_id: meta-llama/Meta-Llama-3-8B
    train_path: /mnt/user_storage/train_data_sample.jsonl
    valid_path: /mnt/user_storage/train_data_sample.jsonl
    context_length: 1024
    num_devices: 4
    num_epochs: 5
    checkpoint_every_n_epochs: 5
    train_batch_size_per_device: 8
    eval_batch_size_per_device: 8
    lr_scheduler_type: constant
    learning_rate: 1e-5
    num_checkpoints_to_keep: 1
    no_gradient_checkpoint: False
    output_dir: /mnt/local_storage
    deepspeed:
      config_path: config_files/deepspeed/zero_3.json
    flash_attention_2: true
    classifier_config:
      label_tokens:
          - "[[1]]"
          - "[[2]]"
          - "[[3]]"
          - "[[4]]"
          - "[[5]]"



```python
# View job yaml config
!cat configs/ft_job.yaml
```

    name: llm-router-tutorial
    entrypoint: python src/ft.py configs/ft_config_debug.yaml
    image_uri: localhost:5555/anyscale/llm-forge:0.5.0.0
    requirements: requirements.txt
    max_retries: 0



```python
import os

# Initialize WANDB API key
os.environ['WANDB_API_KEY'] = '34b8f32abb7ba71277361c99f84d9bea484b5d3b'  # <-- replace with your token

# Job submission
!anyscale job submit --config-file configs/ft_job.yaml --exclude assets
```

    [1m[36mOutput[0m[0m
    [0m[1m[36m(anyscale +1.1s)[0m [0m[0m[0m[0mSubmitting job with config JobConfig(name='llm-router-tutorial', image_uri='localhost:5555/anyscale/llm-forge:0.5.0.0', compute_config=None, env_vars=None, py_modules=None, cloud=None, project=None).[0m
    [0m[1m[36m(anyscale +2.6s)[0m [0m[0m[0m[0mUploading local dir '.' to cloud storage.[0m
    [0m[1m[36m(anyscale +4.0s)[0m [0m[0m[0m[0mJob 'llm-router-tutorial' submitted, ID: 'prodjob_fe4gwqv9spsqzgjf4es3e6r6rk'.[0m
    [0m[1m[36m(anyscale +4.0s)[0m [0m[0m[0m[0mView the job in the UI: https://console.anyscale.com/jobs/prodjob_fe4gwqv9spsqzgjf4es3e6r6rk[0m
    [0m[1m[36m(anyscale +4.0s)[0m [0m[0m[0m[0mUse `--wait` to wait for the job to run and stream logs.[0m
    [0m[0m

TODO: Add some text and screenshot of Ray Dashboard

# Step 3: Offline Evaluation <a id="offline-eval"></a>
TODO: WRITEME


### Load Model from HF
### Evaluate and compare to Random Router



```python
from collections import OrderedDict

router_predictions = OrderedDict()
```


```python
import numpy as np

rng = np.random.RandomState(123)
router_predictions["Random"] = rng.uniform(0, 1, len(subsampled_df))
```


```python
from src.evaluation_metrics import plot_quality_cost_curve

oss_model_scores = subsampled_df["mixtral_score"].to_numpy()
closed_model_scores = np.ones(len(subsampled_df["mixtral_score"])) * 5.0

plot_quality_cost_curve(oss_model_scores, closed_model_scores, router_predictions)
```


    
![png](README_files/README_44_0.png)
    



```python

```
