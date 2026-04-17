"""
Ray Serve entry point.

Local (Anyscale Workspace):
    serve run deploy:app

Production (Anyscale Service):
    anyscale service deploy -f service_config.yaml --working-dir ./
"""
from src.ranker import ProductRanker

app = ProductRanker.bind()
