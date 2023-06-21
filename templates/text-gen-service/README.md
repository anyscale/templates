# Anyscale Service - Text generation

This template provides an example of Anyscale LLM service to  batch inference workload and generate GPT2-outputs from a given prompt!

## Running locally on Anyscale service

Within workspace terminal you can run following command to start the service.

`serve run server:deployment`

Once the service is started succesfully, open another terminal and test the service using following example.

`python query.py "meaning of life is:"`

## Roll out as Anyscale service
After launching the teamplate, simply run the rollowing command from you workspace's terminal.

`anyscale service rollout -f service.yaml`

You should see the terminal output URL to view & manage your service within few seconds. 
