# Cloud Infrastructure Deployment & Disaster Recovery Runbook
Document ID: OPS-RUNBOOK-2026-DR
Target Environment: Production Multi-Region Cluster
Classification: Internal Operations

## 1. Primary Architecture Overview
The production system runs across three geographic cloud regions:
- Primary Region: us-east-1 (Northern Virginia)
- Secondary Hot Standby: eu-central-1 (Frankfurt)
- Disaster Recovery Backup Vault: ap-south-1 (Mumbai)

The ingestion microservices are deployed on Kubernetes (EKS v1.31) with Horizontal Pod Autoscalers (HPA) configured to scale between 10 pods and 80 pods based on 70% average CPU utilization or message backlog depth exceeding 5,000 queue events.

## 2. Disaster Recovery & Failover Protocol
In the event of a catastrophic regional failure in us-east-1:
- DNS Failover: Route53 latency-based routing automatically redirects API traffic to eu-central-1 within 45 seconds of automated health check failure.
- Recovery Time Objective (RTO): The verified RTO for total cluster recovery is 8 minutes.
- Recovery Point Objective (RPO): Database WAL streaming replication guarantees an RPO of less than 15 seconds under normal operational load.
- Failover Command: Automated failover can be manually triggered by on-call site reliability engineers using the CLI command:
  `filewise-ops failover --from us-east-1 --to eu-central-1 --confirm`

## 3. Database Maintenance and Backup Retention
- Automated database snapshots are taken every 4 hours and retained for 30 calendar days.
- Weekly full cold storage archives are encrypted using AES-256 and pushed to AWS Glacier with a 7-year immutable legal hold.
- Database vacuuming and index re-indexing run every Sunday morning between 02:00 UTC and 04:00 UTC during minimal traffic windows.
