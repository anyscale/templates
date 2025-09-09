#!/usr/bin/env python3
"""
Simple TPC-H ETL Demo with Ray Data

This script demonstrates a working ETL pipeline using Ray Data
with clean, simple code that runs reliably on Anyscale.
"""

import ray
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Simple ETL demo that works reliably."""
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # Create large-scale customer data (enterprise size)
    customer_data = []
    logger.info("Generating large-scale customer dataset (this may take a few minutes)...")
    for i in range(50000):  # 50K customers for realistic ETL scale
        customer_data.append({
            'customer_id': i + 1,
            'name': f'Customer_{i+1}',
            'balance': np.random.uniform(1000, 10000),
            'segment': np.random.choice(['Premium', 'Standard', 'Basic']),
            'registration_date': f'2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
            'country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU']),
            'annual_revenue': np.random.uniform(10000, 500000),
            'last_order_days_ago': np.random.randint(1, 365),
            'total_orders': np.random.randint(1, 100)
        })
    
    # Convert to Ray dataset
    customers_ds = ray.data.from_items(customer_data)
    logger.info(f"Created customer dataset: {customers_ds.count()} records")
    
    # Complex transformation for enterprise ETL processing
    def enrich_customer_data(customer):
        balance = customer['balance']
        annual_revenue = customer['annual_revenue']
        last_order_days = customer['last_order_days_ago']
        total_orders = customer['total_orders']
        
        # Calculate customer tier based on multiple factors
        if balance > 7500 and annual_revenue > 200000:
            tier = 'Platinum'
        elif balance > 5000 and annual_revenue > 100000:
            tier = 'Gold'
        elif balance > 2500 or annual_revenue > 50000:
            tier = 'Silver'
        else:
            tier = 'Bronze'
        
        # Calculate customer lifetime value
        avg_order_value = annual_revenue / total_orders if total_orders > 0 else 0
        recency_score = max(0, 1 - (last_order_days / 365))  # Higher score for recent orders
        frequency_score = min(1, total_orders / 100)  # Normalize to 0-1
        monetary_score = min(1, annual_revenue / 500000)  # Normalize to 0-1
        
        clv = (recency_score * 0.3 + frequency_score * 0.4 + monetary_score * 0.3) * annual_revenue
        
        # Risk assessment
        risk_factors = []
        if last_order_days > 180:
            risk_factors.append('inactive')
        if balance < 0:
            risk_factors.append('negative_balance')
        if total_orders < 5:
            risk_factors.append('low_engagement')
        
        risk_level = 'high' if len(risk_factors) >= 2 else 'medium' if len(risk_factors) == 1 else 'low'
        
        return {
            **customer,
            'tier': tier,
            'customer_lifetime_value': clv,
            'avg_order_value': avg_order_value,
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'monetary_score': monetary_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'is_high_value': clv > 100000,
            'is_at_risk': risk_level == 'high',
            'processing_date': '2024-01-01'
        }
    
    # Apply transformation
    logger.info("Starting customer enrichment processing...")
    enriched_customers = customers_ds.map(enrich_customer_data)
    logger.info(f"Enriched customers: {enriched_customers.count()} records")
    
    # Simple aggregation using groupby
    tier_summary = enriched_customers.groupby('tier').count()
    logger.info(f"Tier summary: {tier_summary.count()} groups")
    
    # Additional analysis for enterprise ETL
    high_value_customers = enriched_customers.filter(lambda x: x['is_high_value'])
    at_risk_customers = enriched_customers.filter(lambda x: x['is_at_risk'])
    international_customers = enriched_customers.filter(lambda x: x['country'] != 'US')
    
    logger.info("Performing enterprise-scale analysis...")
    
    # Display comprehensive results
    logger.info("\nEnterprise ETL Results:")
    logger.info("=" * 50)
    
    sample_customers = enriched_customers.take(5)
    for customer in sample_customers:
        logger.info(f"Customer {customer['customer_id']}: {customer['tier']} tier, "
                   f"CLV: ${customer['customer_lifetime_value']:,.2f}, Risk: {customer['risk_level']}")
    
    tier_results = tier_summary.take_all()
    logger.info(f"\nCustomer Tier Distribution:")
    for tier_data in tier_results:
        logger.info(f"  {tier_data['tier']}: {tier_data['count()']:,} customers")
    
    # Business insights
    logger.info(f"\nBusiness Intelligence Insights:")
    logger.info(f"  Total customers processed: {enriched_customers.count():,}")
    logger.info(f"  High-value customers: {high_value_customers.count():,}")
    logger.info(f"  At-risk customers: {at_risk_customers.count():,}")
    logger.info(f"  International customers: {international_customers.count():,}")
    
    # Calculate percentages
    total_count = enriched_customers.count()
    if total_count > 0:
        high_value_pct = (high_value_customers.count() / total_count) * 100
        at_risk_pct = (at_risk_customers.count() / total_count) * 100
        intl_pct = (international_customers.count() / total_count) * 100
        
        logger.info(f"\nKey Metrics:")
        logger.info(f"  High-value customer rate: {high_value_pct:.1f}%")
        logger.info(f"  At-risk customer rate: {at_risk_pct:.1f}%")
        logger.info(f"  International customer rate: {intl_pct:.1f}%")
    
    logger.info("\nEnterprise ETL pipeline completed successfully!")

if __name__ == "__main__":
    main()
