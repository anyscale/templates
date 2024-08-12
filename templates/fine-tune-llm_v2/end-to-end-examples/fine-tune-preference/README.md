# Preference Tuning for Summarization using Synthetic Data

**⏱️ Time to complete**: \<TODO\>

Preference tuning is a powerful tool that can optimize LLMs towards complex preferences that can not easily captured through supervised fine-tuning. However, manually annotating preferences between model outputs using human raters can be extremely time-consuming and expensive. Here we'll go through an end-to-end example for preference tuning of an open-source language model, covering data preprocessing, fine-tuning and evaluation. 

We will focus on the task of summarization for the [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. 

# Table of Contents
1. [Data Preprocessing](#step-1-data-preprocessing): In this section we cover how we can prepare preference data for the summarization task using an LLM-as-a-judge. 
2. [DPO Finetuning](#step-2-fine-tuning): This section will cover how you can fine-tune an open source model on the preference data on the Anyscale platform.
3. [Evaluation](#step-3-evaluation): The section will lay down a blue-print for evaluation and compare performance to that of closed source models like OpenAI's GPT-4.
4. [Iterative-DPO](#step-4-iterative): An optional step to further boost performance with iterative preference-tuning. 

First, let's make the necessary imports


```python
import os
import yaml
import datasets
import openai

import ray.data
```

# Step 1: Synthetic Data Generation

First, let's inspect the training dataset and look at an example. 


```python
hf_ds = datasets.load_dataset("abisee/cnn_dailymail", '3.0.0', split="train").shuffle(seed=21)
# extract a subset of 20000 articles
hf_ds_subset =  hf_ds.select(range(20000))

ray_ds = ray.data.from_huggingface(hf_ds_subset)
raw_example = ray_ds.take(1)[0]
```


    Downloading readme:   0%|          | 0.00/15.6k [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/257M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/257M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/259M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/34.7M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/30.0M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/287113 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/13368 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/11490 [00:00<?, ? examples/s]



    Downloading readme:   0%|          | 0.00/15.6k [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/257M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/257M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/259M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/34.7M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/30.0M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/287113 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/13368 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/11490 [00:00<?, ? examples/s]


    2024-08-09 11:11:41,594	INFO worker.py:1596 -- Connecting to existing Ray cluster at address: 10.0.4.151:6379...
    2024-08-09 11:11:41,601	INFO worker.py:1772 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-m4a38rehf7miww178mefsrumy2.i.anyscaleuserdata.com [39m[22m
    2024-08-09 11:11:41,603	INFO packaging.py:358 -- Pushing file package 'gcs://_ray_pkg_571a453227fe1f71a0db8d4c7877fab901d9fc29.zip' (0.10MiB) to Ray cluster...
    2024-08-09 11:11:41,604	INFO packaging.py:371 -- Successfully pushed file package 'gcs://_ray_pkg_571a453227fe1f71a0db8d4c7877fab901d9fc29.zip'.
    2024-08-09 11:11:49,781	INFO dataset.py:2416 -- Tip: Use `take_batch()` instead of `take() / show()` to return records in pandas or numpy batch format.
    2024-08-09 11:11:49,784	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-08-09_10-34-55_275931_2296/logs/ray-data
    2024-08-09 11:11:49,785	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> LimitOperator[limit=1]



    - limit=1 1: 0 bundle [00:00, ? bundle/s]



    Running 0: 0 bundle [00:00, ? bundle/s]


    [36m(autoscaler +1h53m32s)[0m Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.



```python
import pprint 
pprint.pprint(raw_example, width=100)
```

    {'article': 'Scam: Lisa Harrison, 34, promised customers low currency rates on US dollars and '
                'special deals . A wedding planner who stole £80,000 from couples in a bid to satisfy '
                "an 'out-of-control' online gambling addiction has been jailed. Lisa Harrison, 34, "
                'began taking money from her clients in summer 2013 by enticing them with low currency '
                'rates on US dollars and flight upgrades. She took money from 19 couples who had '
                'entrusted their savings to her after being promised the wedding of their dreams. It '
                'is understood that the company she worked for, iPlan New York, specialised in '
                'weddings in New York City. Her website iplannewyork.com, which has been taken down, '
                "said: 'iPlan New York was set up to create and style the perfect tailor made wedding "
                "for couples travelling to New York to get married! 'We are passionate about what we "
                'do and passionate about New York! We have experience in planning NYC weddings for '
                "couples from all over the world.' But she was arrested in December last year after "
                'eventually coming clean to her victims in an email and saying she had been forced to '
                'close her business. Police soon found she had taken £80,107 from the couples and '
                'spent a staggering £77,933 on gambling sites Paddy Power and William Hill. The '
                "business' Facebook page has also been deleted, but outraged victims have shared their "
                "victims on a wedding forum. One victim called Jennifer wrote in November 2013: 'I had "
                'previously given Lisa a positive review because our vow renewal went wonderful. '
                "'Little did I know until last week that she didn't even pay the vendors that helped "
                "with our ceremony. 'I am so disgusted and can't fathom such an act. We paid her in "
                "full and to think that our photographer didn't even get paid is just astonishing to "
                "me. 'I feel so horrible for the other couples that had their perfect day planned and "
                "this woman decided to perform such an act. 'I pray for each of you in hopes that you "
                'will be able to move on from this and live a healthy and happy life with your '
                "significant other. 'I can't believe this woman took our money and did such an "
                "unthinkable act. God bless all of you and I hope this mess gets corrected quickly.' "
                'While another anonymous victim posted a copy of the email they claim they had been '
                "sent by Harrison when she admitted the scam. It read: 'I have to announce the closure "
                "of iPlan New York. 'For some time now I have been battling against a gambling "
                'addiction that has seen me lose all of the company money including money paid to me '
                "by you for services and dollars. 'I cannot go on another day with this situation as "
                'this illness has taken me over completely and I have to both face up to the '
                "consequences of my actions and seek help for the debilitating addiction. 'I am "
                "extremely ill with it and need to seek help as soon as possible. 'I am completely "
                'devastated that not only have I lost money of yours but betrayed your trust as a '
                "wedding planner. 'Right now I am uncertain as to what the future holds with regards "
                'to future weddings already planned, I will be in touch with the suppliers in NYC to '
                'inform them also. Sentence: Harrison, of Earith, Cambridgeshire, was jailed for two '
                "years at Peterborough Crown Court, above . 'I will today be having my computer and "
                'all electronic devices ceased (sic) under an intervention and handing myself into the '
                "police to give a statement and to tell them everything. 'No doubt you will be "
                'informing the police too and for those purposes it will be the Cambridgeshire '
                'Constabulary and my full name is Lisa Harrison and I will be handing myself in after '
                "sending these emails. 'I won’t be able to reply to any emails or calls for the time "
                "being as I will not have access. 'I am truly from the bottom of my heart so sorry for "
                'everything, as with addictions I thought I had everything under control and was in '
                'denial that I could put everything right, which I have been trying so desperately to '
                "do. 'As soon as and if I am able to communicate further about any outstanding issues "
                "I will do so. Lisa.' Posting on an online review site for the wedding service, one "
                "former customer said: 'We are due to go in less than 48 hours and we have nothing!! "
                "She has now closed down her website too! She has left us devastated!' Another, using "
                "the name Shaun, wrote: 'Alarm bells rang for me when she asked for all our spending "
                'money cos she had a deal on a currency card. While one woman, using the name Andrea, '
                "said: 'I am absolutely devastated for anyone who has used iPlan New York and "
                "subsequently been let down'. She added that she had a 'gut feeling' not to pay "
                'upfront. Harrison, of Earith, Cambridgeshire, admitted fraudulent trading and was '
                'jailed for two years at Peterborough Crown Court on Tuesday. Det Sgt Iain Moor, from '
                "Cambridgeshire Constabulary, said: 'This was an extremely distressing case for the 19 "
                "couples who lost life savings and had their dream day ruined by Harrison. 'I hope the "
                'victims received some comfort in the prison sentence imposed on Harrison, meaning '
                "they can now start to re-build their lives.'",
     'highlights': 'Lisa Harrison enticed clients with low currency rates and flight upgrades .\n'
                   'She took money from 19 couples who were promised dream weddings .\n'
                   'Harrison spent nearly £78,000 on gambling sites including Paddy Power .\n'
                   'She admitted the scam to victims in an email before handing herself in .\n'
                   "Outraged victims say they are 'disgusted' and have been left 'devastated'\n"
                   "In email Harrison says she will 'seek help for the debilitating addiction'\n"
                   'She admitted fraudulent trading and was jailed for two years on Tuesday .',
     'id': '4feb82c680166f0b8f90bf3a6f9779b04f229325'}


Note that we currently only have raw articles and we need to finally get to preference data for summaries i.e summaries with some notion of like/dislike. Traditionally, this would involve generating summaries using the base model you wish to fine-tune and asking human annotators to provide a rating for each sample. In this example, we will employ a synthetic reward model i.e we will ask a language model to judge generated responses from the base model. For this example, we will use `Mistral-7B-Instruct-v0.1` as the base model to fine-tune and `Llama-3-70B-Instruct` as a judge

Our data pre-processing is going to look as follows: 

![preprocessing](./assets/preprocessing.png?1)

# TODO: Instructions for pre-processing
\<Provide a better descrption for the data preprocessing and the choices made.\>

\<We have the relevant preprocessing code in `utils/generate_questions.py` and `utils/generate_summaries_and_scores.py`. You can run data generation as an Anyscale job with configs/generate_questions_job.yaml and configs/generate_summaries_job.yaml.\>

\<After preprocessing, here's an example for the Q&A generated by Llama 70B and here's an example for the summaries generated by Mistral 7B Instruct \>


\<We sample chosen and rejected messages from the summaries based on the Q&A Accuracy score. We use a threshold of 3/5 for classifying examples as 'chosen' and 'rejected'. Here's an example training dataset sample for the DPO model\>

# Step 2: Fine-tuning

Now that we have the pre-processed dataset, we are ready to fine-tune `Mistral-7B-Instruct-v0.1` using DPO. On Anyscale, we've created an easy-to-use interface to do preference-tuning using `DPO`. We leverage Ray to overlap reference model log-probability calculation with model training to improve GPU utilization. Most implementations compute log probabilities synchronously with model training,

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm_v2/end-to-end-examples/fine-tune-preference/assets/hf_dpo.png"/>

While our implementation using Ray is asynchronous:  


<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm_v2/end-to-end-examples/fine-tune-preference/assets/anyscale_dpo.png"/>

Further, our use of Ray Data also implies that the compute configuration for the reference model can be completely decoupled with the policy model. For example, reference model calculation can run on a different node with zero code changes needed. 


To get started with DPO training, we provide the config for DPO in [configs/mistral_dpo_summarization.yaml](configs/mistral_dpo_summarization.yaml) . 


TODO: The provided config uses 6 and 2 A10s and doesn't utilize GPUs properly. We should improve logprob processor


```python
!cat configs/mistral_dpo_summarization.yaml
```

    model_id: mistralai/Mistral-7B-Instruct-v0.1
    # Example summarization dataset with 10k examples for training with an average of 2.2k tokens per sample
    train_path: s3://air-example-data/preference-tuning-summarization/train.jsonl
    valid_path: s3://air-example-data/preference-tuning-summarization/valid.jsonl
    task: "preference_tuning"
    context_length: 4096
    # For DPO, it is recommended to set a high `num_data_blocks_per_device` to not bottleneck the logp processor.
    # We recommend not going beyond 20 so as to not spawn too many Ray actors. 
    num_data_blocks_per_device: 16
    num_devices: 6 # <--- runs training on 6 GPUs
    train_batch_size_per_device: 2
    eval_batch_size_per_device: 2
    learning_rate: 5e-6
    num_epochs: 3
    no_gradient_checkpoint: False
    output_dir: /mnt/local_storage/
    deepspeed:
      config_path: deepspeed_configs/zero_3.json
    worker_resources:
      accelerator_type:A10G: 1
    flash_attention_2: True
    padding: "longest"
    preference_tuning_config:
      beta: 0.01
      logprob_processor_scaling_config:
        custom_resources:
          accelerator_type:A10G: 1
        concurrency: 2 # <--- runs reference model logp calculation on 2 GPUs
        batch_size: 2
    lora_config:
      r: 8
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
      modules_to_save: []
      bias: "none"
      fan_in_fan_out: false
      init_lora_weights: true


```python
!llmforge anyscale finetune end-to-end-examples/fine-tune-preference/configs/mistral_dpo_summarization.yaml
```

# Step 3: Evaluation

Let's evaluate our trained model. Here we'll use two baselines: (1) the base model before finetuning (reference model in DPO) and (2) GPT-4o.

## Evaluation strategy

Our evaluation strategy involves the same Q&A scoring system as used while generating the preference data.



\<TODO: Add a nice diagram similar to data preprocessing, but just for the evaluation flow \>


\<TODO: Add description\>



# Step 4: Iterative-DPO (optional)

TODO
