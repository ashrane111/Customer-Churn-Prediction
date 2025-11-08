"""Configure Datadog monitoring."""
import os
from datadog import initialize, api

# Initialize Datadog
initialize(
    api_key=os.getenv('DATADOG_API_KEY'),
    app_key=os.getenv('DATADOG_APP_KEY')
)

# Create monitors
monitors = [
    {
        "name": "Churn API - High Latency",
        "type": "metric alert",
        "query": "avg(last_5m):p95:trace.http.request{service:churn-api} > 0.2",
        "message": "API P95 latency exceeded 200ms! @pagerduty",
        "tags": ["service:churn-api", "team:mlops"],
        "options": {
            "thresholds": {"critical": 0.2, "warning": 0.15},
            "notify_no_data": True,
            "no_data_timeframe": 10
        }
    },
    {
        "name": "Model Drift Detected",
        "type": "metric alert",
        "query": "avg(last_1h):avg:custom.model.drift.psi{} > 0.15",
        "message": "PSI drift score > 0.15. Model may need retraining.",
        "tags": ["ml:drift", "service:churn-api"],
        "options": {
            "thresholds": {"critical": 0.15, "warning": 0.1}
        }
    },
    {
        "name": "Low Model Performance",
        "type": "metric alert",
        "query": "avg(last_1d):avg:custom.model.auc{} < 0.83",
        "message": "Model AUC dropped below threshold. Investigate immediately.",
        "tags": ["ml:performance", "service:churn-api"]
    },
    {
        "name": "High Error Rate",
        "type": "metric alert",
        "query": "avg(last_5m):sum:trace.http.request.errors{service:churn-api}.as_rate() > 0.02",
        "message": "Error rate > 2%. Check logs for details.",
        "tags": ["service:churn-api", "severity:high"]
    },
    {
        "name": "ECS Task Failures",
        "type": "metric alert",
        "query": "sum(last_5m):aws.ecs.service.running{service:churn-api} < 1",
        "message": "No running ECS tasks! Service is down!",
        "tags": ["infrastructure", "critical"]
    }
]

for monitor in monitors:
    api.Monitor.create(**monitor)
    print(f"Created monitor: {monitor['name']}")

# Create dashboard
dashboard = {
    "title": "Churn Prediction MLOps",
    "description": "Production monitoring for churn prediction service",
    "widgets": [
        {
            "definition": {
                "title": "API Latency (P50, P95, P99)",
                "type": "timeseries",
                "requests": [
                    {
                        "q": "p50:trace.http.request{service:churn-api}",
                        "display_type": "line",
                        "style": {"palette": "green"}
                    },
                    {
                        "q": "p95:trace.http.request{service:churn-api}",
                        "display_type": "line",
                        "style": {"palette": "yellow"}
                    },
                    {
                        "q": "p99:trace.http.request{service:churn-api}",
                        "display_type": "line",
                        "style": {"palette": "red"}
                    }
                ]
            }
        },
        {
            "definition": {
                "title": "Request Rate",
                "type": "query_value",
                "requests": [{
                    "q": "sum:trace.http.request{service:churn-api}.as_rate()",
                    "aggregator": "avg"
                }],
                "precision": 0
            }
        },
        {
            "definition": {
                "title": "Model Drift (PSI)",
                "type": "timeseries",
                "requests": [{
                    "q": "avg:custom.model.drift.psi{}",
                    "display_type": "area"
                }],
                "markers": [
                    {"value": "y = 0.1", "display_type": "warning dashed"},
                    {"value": "y = 0.15", "display_type": "error dashed"}
                ]
            }
        },
        {
            "definition": {
                "title": "Model Performance",
                "type": "timeseries",
                "requests": [
                    {"q": "avg:custom.model.auc{}", "display_type": "line"},
                    {"q": "avg:custom.model.f1{}", "display_type": "line"},
                    {"q": "avg:custom.model.precision{}", "display_type": "line"}
                ]
            }
        },
        {
            "definition": {
                "title": "ECS Resource Utilization",
                "type": "heatmap",
                "requests": [{
                    "q": "avg:aws.ecs.cpuutilization{service:churn-api} by {task}"
                }]
            }
        },
        {
            "definition": {
                "title": "Error Logs",
                "type": "log_stream",
                "logset": "main",
                "query": "service:churn-api status:error"
            }
        }
    ],
    "layout_type": "ordered",
    "notify_list": ["@slack-mlops-alerts"]
}

api.Dashboard.create(**dashboard)
print("Dashboard created successfully")