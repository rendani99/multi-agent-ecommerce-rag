# NovaMart Multi-Agent Customer Support System (Amazon Bedrock AgentCore & Strands SDK)

## Overview & Architecture
This project implements a production-grade multi-agent customer support platform for NovaMart using the Orchestrator -> Workers pattern with the Strands Agents SDK and Amazon Bedrock AgentCore.

### Key Components:
1. **OrchestratorAgent (Claude 4.5 Haiku, temp=0.0)**: Central routing engine managing state in DynamoDB `WorkflowState` with optimistic locking and strict 6-rule routing. Never directly responds to customers.
2. **InventoryAgent (Claude 4.5 Sonnet, temp=0.1)**: Factual data gatherer querying `udacity-agentcore-orders` and `udacity-agentcore-customers`.
3. **PolicyAgent (Claude 4.5 Sonnet, temp=0.2)**: Multi-agent RAG coordinator running three specialized retriever sub-agents (Returns, Shipping, Warranty) in parallel using `ThreadPoolExecutor(max_workers=3)`.
4. **RefundAgent (Claude 4.5 Sonnet, temp=0.1)**: Enforces return policies (30 days for Standard customers, 60 days for Premium customers) and updates order status.
5. **CommunicationAgent (Claude 4.5 Sonnet, temp=0.3)**: Inspects the complete workflow state and crafts empathetic, customer-ready messages.
6. **Enterprise Guardrail**: Blocks PII (credit cards, SSNs), anonymizes contact data, denies competitor mentions and haggling, and applies content moderation.
7. **AgentCore Memory**: Session summary strategy with 7-day event retention.
8. **Distributed Observability**: 100% X-Ray tracing across every worker node and CloudWatch INFO logging.

## Submission Deliverables Included:
- `src/agent_orchestrator.py` — Fully implemented (Tasks 2, 3, 4, 6)
- `.env` — Populated with KB IDs, Guardrail ID/Version, and AgentCore Runtime ARN
- `test_results_120_score.png` — Screenshot showing 120/120 points passed across all rubric tasks
- `xray_service_map_trace.png` — AWS X-Ray Service Map screenshot demonstrating end-to-end distributed trace graph
