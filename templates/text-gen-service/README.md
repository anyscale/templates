# Anyscale Service - Text generation

This template provides an example of Anyscale LLM Service to generate GPT2-outputs from a given prompt.

## Running locally on an Anyscale Workspace

After launching this template, within the Workspace terminal you can run following command to start the Service.

`serve run server:deployment`

Once the service is started successfully, open another terminal and test the Service using following example.

`python query.py "meaning of life is:"`

## Roll out as Anyscale service

After launching the teamplate, open `service.yaml` file and change `name` field to give your Service a name. Then just run following command to launch a Anyscale Service.

`anyscale service rollout -f service.yaml`

You should see the terminal output URL to view & manage your Service within few seconds. 
