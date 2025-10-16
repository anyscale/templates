
# Tracing Guide

Deploy the service using the following command.

```
anyscale service deploy -f default_tracing_service.yaml
```

After querying your application, Anyscale exports traces to the `/tmp/ray/session_latest/logs/serve/spans/` folder on instances with active replicas.

Sample traces are shown in the directory after a simple GET request to the service.
- `proxy_10.0.56.31_tracing.jsonl`
- `replica_default_tracing_HelloWorld_i852aab6_tracing.jsonl`



For the full tracing guide, see [this docs page](https://docs.anyscale.com/monitoring/tracing/).


