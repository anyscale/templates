# Anyscale Service - Text generation

This template provides an example of Anyscale LLM service to generate GPT2-outputs from a given prompt.

## Running locally on Anyscale service

After launching this template, within the Workspace terminal you can run following command to start the service.

`serve run server:deployment`

Once the service is started succesfully, open another terminal and test the service using following example.

`python query.py "meaning of life is:"`

## Roll out as Anyscale service

After launching the teamplate, open `service.yaml` file and change `name` field to give your service a name. Then just run following command to launch a anyscale service.

`anyscale service rollout -f service.yaml`

You should see the terminal output URL to view & manage your service within few seconds. 
