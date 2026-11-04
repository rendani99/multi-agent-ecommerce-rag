"""
NovaMart Multi-Agent Customer Support System
Orchestrator -> Workers hierarchy built with Amazon Bedrock AgentCore and Strands Agents SDK.
"""
import os
import sys
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import boto3
from strands_agents import Agent
# Note: @tool from agent_observability provides automatic X-Ray subsegments
from src.agent_observability import tool
import config
from src.bedrock_kb_retrieval import retrieve_from_knowledge_base
from src.agent_utils import AgentTrace
import src.agentcore_cli as agentcore_cli

logger = logging.getLogger(__name__)

dynamodb = boto3.resource("dynamodb", region_name=config.AWS_REGION)
bedrock = boto3.client("bedrock", region_name=config.AWS_REGION)
agentcore_control = boto3.client("bedrock-agentcore-control", region_name=config.AWS_REGION)

# =====================================================================
# WorkflowState Helpers with Optimistic Locking
# =====================================================================
def _create_workflow_state(session_id: str, customer_id: str) -> Dict[str, Any]:
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    state = {
        "session_id": session_id,
        "customer_id": customer_id,
        "version": 1,
        "inventory_context": {},
        "policy_context": {},
        "refund_decision": {},
        "final_response": ""
    }
    table.put_item(Item=state)
    return state


def _read_workflow_state(session_id: str) -> Dict[str, Any]:
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    res = table.get_item(Key={"session_id": session_id})
    return res.get("Item", {})


def _update_workflow_state(session_id: str, updates: Dict[str, Any], expected_version: int) -> Dict[str, Any]:
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    update_expr = []
    expr_attr_names = {}
    expr_attr_vals = {
        ":inc": 1,
        ":expected": expected_version
    }

    for k, v in updates.items():
        attr_name = f"#{k}"
        attr_val = f":{k}"
        update_expr.append(f"{attr_name} = {attr_val}")
        expr_attr_names[attr_name] = k
        expr_attr_vals[attr_val] = v

    update_expr.append("#version = #version + :inc")
    expr_attr_names["#version"] = "version"

    res = table.update_item(
        Key={"session_id": session_id},
        UpdateExpression="SET " + ", ".join(update_expr),
        ConditionExpression="#version = :expected",
        ExpressionAttributeNames=expr_attr_names,
        ExpressionAttributeValues=expr_attr_vals,
        ReturnValues="ALL_NEW"
    )
    return res.get("Attributes", {})


# =====================================================================
# 1.A — InventoryAgent Worker
# =====================================================================
def build_inventory_agent() -> Agent:
    """Builds the InventoryAgent to gather customer and order facts from DynamoDB."""
    orders_table = dynamodb.Table(config.ORDERS_TABLE)
    customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)

    @tool
    def check_order_status(customer_id: str, order_id: str) -> str:
        """
        Retrieves order details from DynamoDB using customer_id and order_id.
        Returns order status, purchase date, item summary, and total amount.
        """
        try:
            res = orders_table.get_item(Key={"customer_id": customer_id, "order_id": order_id})
            item = res.get("Item")
            if not item:
                return f"Order {order_id} not found for customer {customer_id}."
            return json.dumps(item, default=str)
        except Exception as e:
            return f"Error fetching order: {str(e)}"

    @tool
    def get_customer_tier(customer_id: str) -> str:
        """
        Retrieves the customer tier (Standard or Premium) and account details from DynamoDB.
        """
        try:
            res = customers_table.get_item(Key={"customer_id": customer_id})
            item = res.get("Item")
            if not item:
                return f"Customer {customer_id} not found."
            return json.dumps(item, default=str)
        except Exception as e:
            return f"Error fetching customer: {str(e)}"

    @tool
    def list_customer_orders(customer_id: str) -> str:
        """
        Lists recent orders placed by a specific customer.
        """
        try:
            res = orders_table.query(
                KeyConditionExpression="customer_id = :cid",
                ExpressionAttributeValues={":cid": customer_id}
            )
            items = res.get("Items", [])
            return json.dumps(items, default=str)
        except Exception as e:
            return f"Error listing orders: {str(e)}"

    prompt = (
        "You are the NovaMart Inventory Specialist Agent. Your sole responsibility is gathering "
        "order and customer facts from DynamoDB. You must report factual data accurately. "
        "Never make return or refund decisions yourself."
    )

    return Agent(
        model=config.WORKER_MODEL_ID,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
        system_prompt=prompt,
        temperature=0.1
    )


# =====================================================================
# 1.B — RefundAgent Worker
# =====================================================================
def build_refund_agent(session_id: str) -> Agent:
    """Builds the RefundAgent to evaluate refund eligibility based on order age and customer tier."""
    orders_table = dynamodb.Table(config.ORDERS_TABLE)

    @tool
    def get_inventory_context() -> str:
        """
        Reads order and customer facts gathered by InventoryAgent from the shared WorkflowState.
        """
        state = _read_workflow_state(session_id)
        return json.dumps(state.get("inventory_context", {}), default=str)

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> str:
        """
        Updates the order status to REFUNDED in DynamoDB once eligibility is confirmed.
        """
        try:
            orders_table.update_item(
                Key={"customer_id": customer_id, "order_id": order_id},
                UpdateExpression="SET #status = :s, refund_reason = :r",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={":s": "REFUNDED", ":r": reason}
            )
            return f"Refund initiated successfully for order {order_id}."
        except Exception as e:
            return f"Failed to initiate refund: {str(e)}"

    prompt = (
        "You are the NovaMart Refund Specialist Agent. You make refund eligibility decisions. "
        "First call get_inventory_context to review the customer's tier and order purchase date. "
        "Policy rules: Standard tier customers have a 30-day return window. "
        "Premium tier customers have a 60-day return window. "
        "If eligible, call initiate_refund. Return a clear explanation of your decision."
    )

    return Agent(
        model=config.WORKER_MODEL_ID,
        tools=[get_inventory_context, initiate_refund],
        system_prompt=prompt,
        temperature=0.1
    )


# =====================================================================
# 1.C — PolicyAgent (Multi-Agent Parallel RAG)
# =====================================================================
def build_policy_agent() -> Agent:
    """Builds the PolicyAgent coordinator that fans out parallel queries to 3 retriever sub-agents."""
    
    # 1. Returns Policy Retriever Sub-Agent
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieves passages from the Returns Policy Knowledge Base."""
        return retrieve_from_knowledge_base(config.RETURNS_KB_ID, query)

    returns_agent = Agent(
        model=config.WORKER_MODEL_ID,
        tools=[retrieve_returns_policy],
        system_prompt="Retrieve returns policy terms and conditions.",
        temperature=0.0
    )

    # 2. Shipping Policy Retriever Sub-Agent
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieves passages from the Shipping Policy Knowledge Base."""
        return retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query)

    shipping_agent = Agent(
        model=config.WORKER_MODEL_ID,
        tools=[retrieve_shipping_policy],
        system_prompt="Retrieve shipping policy terms, rates, and transit times.",
        temperature=0.0
    )

    # 3. Warranty Policy Retriever Sub-Agent
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieves passages from the Warranty Policy Knowledge Base."""
        return retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query)

    warranty_agent = Agent(
        model=config.WORKER_MODEL_ID,
        tools=[retrieve_warranty_policy],
        system_prompt="Retrieve product warranty and replacement terms.",
        temperature=0.0
    )

    @tool
    def search_all_policies(query: str) -> str:
        """
        Executes parallel multi-agent RAG across Returns, Shipping, and Warranty Knowledge Bases
        using a ThreadPoolExecutor.
        """
        results = {}
        retrievers = {
            "returns": (returns_agent, f"Find return policies regarding: {query}"),
            "shipping": (shipping_agent, f"Find shipping policies regarding: {query}"),
            "warranty": (warranty_agent, f"Find warranty policies regarding: {query}")
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_domain = {
                executor.submit(agent.run, q): domain
                for domain, (agent, q) in retrievers.items()
            }
            for future in as_completed(future_to_domain):
                domain = future_to_domain[future]
                try:
                    res = future.result()
                    results[domain] = str(res)
                except Exception as e:
                    results[domain] = f"Error querying {domain}: {str(e)}"

        return json.dumps(results, indent=2)

    prompt = (
        "You are the NovaMart Policy Specialist Coordinator. When asked any policy question, "
        "always call search_all_policies to gather context across all three Knowledge Bases in parallel. "
        "Synthesize the retrieved passages into a comprehensive, grounded policy answer."
    )

    return Agent(
        model=config.WORKER_MODEL_ID,
        tools=[search_all_policies],
        system_prompt=prompt,
        temperature=0.2
    )


# =====================================================================
# 1.D — CommunicationAgent Worker
# =====================================================================
def build_communication_agent(session_id: str) -> Agent:
    """Builds the CommunicationAgent to draft empathetic, customer-ready responses."""
    @tool
    def get_full_workflow_context() -> str:
        """Reads the complete WorkflowState record containing findings from all worker agents."""
        state = _read_workflow_state(session_id)
        return json.dumps(state, default=str)

    prompt = (
        "You are the NovaMart Customer Communication Specialist. Call get_full_workflow_context "
        "to inspect all gathered order details, policy context, or refund decisions. "
        "Compose an empathetic, warm, clear, and professional response directly addressing the customer. "
        "Do not output raw JSON."
    )

    return Agent(
        model=config.WORKER_MODEL_ID,
        tools=[get_full_workflow_context],
        system_prompt=prompt,
        temperature=0.3
    )


# =====================================================================
# 1.E — OrchestratorAgent Coordinator
# =====================================================================
def build_orchestrator_agent(session_id: str, customer_id: str) -> Agent:
    """Builds the central OrchestratorAgent that routes requests and manages WorkflowState."""
    
    @tool
    def initialize_session() -> str:
        """Initializes the shared DynamoDB WorkflowState for the current session."""
        _create_workflow_state(session_id, customer_id)
        return f"Session {session_id} initialized for customer {customer_id}."

    @tool
    def route_to_inventory_agent(task_prompt: str) -> str:
        """Routes order or account fact-gathering requests to the InventoryAgent."""
        state = _read_workflow_state(session_id)
        expected_v = state.get("version", 1)
        agent = build_inventory_agent()
        result = str(agent.run(task_prompt))
        _update_workflow_state(session_id, {"inventory_context": {"findings": result}}, expected_v)
        return f"Inventory agent completed: {result}"

    @tool
    def route_to_policy_agent(policy_query: str) -> str:
        """Routes policy questions to the PolicyAgent for parallel multi-KB retrieval."""
        state = _read_workflow_state(session_id)
        expected_v = state.get("version", 1)
        agent = build_policy_agent()
        result = str(agent.run(policy_query))
        _update_workflow_state(session_id, {"policy_context": {"findings": result}}, expected_v)
        return f"Policy agent completed: {result}"

    @tool
    def route_to_refund_agent(refund_prompt: str) -> str:
        """Routes return/refund requests to the RefundAgent to evaluate eligibility."""
        state = _read_workflow_state(session_id)
        expected_v = state.get("version", 1)
        agent = build_refund_agent(session_id)
        result = str(agent.run(refund_prompt))
        _update_workflow_state(session_id, {"refund_decision": {"findings": result}}, expected_v)
        return f"Refund agent completed: {result}"

    @tool
    def route_to_communication_agent(instructions: str) -> str:
        """Routes to the CommunicationAgent to compose the final customer response."""
        state = _read_workflow_state(session_id)
        expected_v = state.get("version", 1)
        agent = build_communication_agent(session_id)
        final_msg = str(agent.run(instructions))
        _update_workflow_state(session_id, {"final_response": final_msg}, expected_v)
        return final_msg

    system_prompt = (
        "You are the NovaMart Master Support Orchestrator. Follow these strict routing rules:\n"
        "1. Every request, always: Call initialize_session first.\n"
        "2. Order status / return / refund requests: Route to inventory agent first, then refund agent.\n"
        "3. Policy meaning questions (return windows, shipping rates, warranty terms): Route to policy agent.\n"
        "4. Account questions ('what is my tier?', 'am I premium?'): Route to inventory agent — never policy agent.\n"
        "5. Math / calculation questions: Orchestrator -> CommunicationAgent, skip Inventory, Policy and Refund.\n"
        "6. Every request, always (last step): Route to communication agent to compose the final reply.\n"
        "CRITICAL: You must NEVER compose the final customer response yourself. Always delegate "
        "to route_to_communication_agent as your final tool call."
    )

    return Agent(
        model=config.ORCHESTRATOR_MODEL_ID,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent
        ],
        system_prompt=system_prompt,
        temperature=0.0
    )


# =====================================================================
# TASK 3: Bedrock Guardrail and AgentCore Runtime Deployment
# =====================================================================
def create_guardrail() -> tuple[str, str]:
    """Creates a Bedrock Guardrail with enterprise content, PII, topic, and word policies."""
    res = bedrock.create_guardrail(
        name="novamart-enterprise-guardrail",
        description="Guardrail enforcing enterprise safety, PII protection, and topic boundaries.",
        topicPolicyConfig={
            "tierConfig": {"tierName": "STANDARD"},
            "topicsConfig": [
                {
                    "name": "competitor_products",
                    "definition": "Questions or statements referencing direct competitor e-commerce platforms or stores.",
                    "type": "DENY"
                },
                {
                    "name": "pricing_negotiations",
                    "definition": "Haggling or requesting discounts/price changes beyond advertised catalog pricing. Calculations using standard advertised prices and discount percentages are permitted.",
                    "type": "DENY"
                },
                {
                    "name": "legal_threats",
                    "definition": "Threats of legal action, lawsuits, or regulatory complaints against NovaMart.",
                    "type": "DENY"
                }
            ]
        },
        contentPolicyConfig={
            "filtersConfig": [
                {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                {"type": "MISCONDUCT", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"}
            ]
        },
        wordPolicyConfig={
            "managedWordListsConfig": [{"type": "PROFANITY"}]
        },
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
                {"type": "EMAIL", "action": "ANONYMIZE"},
                {"type": "PHONE", "action": "ANONYMIZE"}
            ]
        },
        blockedInputMessaging="I apologize, but your request violated our safety policies and cannot be processed.",
        blockedOutputsMessaging="I apologize, but the response could not be displayed due to safety standards.",
        crossRegionConfig={"guardrailProfileIdentifier": "us.guardrail.v1:0"}
    )
    guardrail_id = res["guardrailId"]
    ver_res = bedrock.create_guardrail_version(guardrailIdentifier=guardrail_id, description="Production v1")
    guardrail_version = ver_res["version"]
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime() -> str:
    """Configures and deploys the multi-agent application to AgentCore Runtime via AgentCore CLI."""
    env_vars = {
        "AWS_REGION": config.AWS_REGION,
        "PROJECT_NAME": config.PROJECT_NAME,
        "RETURNS_KB_ID": config.RETURNS_KB_ID,
        "SHIPPING_KB_ID": config.SHIPPING_KB_ID,
        "WARRANTY_KB_ID": config.WARRANTY_KB_ID,
        "AGENT_LOG_GROUP": config.AGENT_LOG_GROUP,
        "GUARDRAIL_ID": config.GUARDRAIL_ID,
        "GUARDRAIL_VERSION": config.GUARDRAIL_VERSION
    }
    agentcore_cli.configure_runtime(
        network_mode="PUBLIC",
        protocol="HTTP",
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
        env_vars=env_vars
    )
    agentcore_cli.deploy()
    return agentcore_cli.deployed_runtime_arn()


# =====================================================================
# TASK 4: AgentCore Memory Configuration
# =====================================================================
def configure_memory(runtime_arn: str) -> str:
    """Configures an AgentCore Memory resource with a summary memory strategy and 7-day retention."""
    res = agentcore_control.create_memory(
        name="NovaMartCustomerSupportMemory",
        description="Session summary memory for NovaMart multi-turn customer support conversations.",
        eventExpiryDuration=7,
        memoryStrategies=[
            {
                "summaryMemoryStrategy": {
                    "name": "customer_session_summary"
                }
            }
        ]
    )
    memory_arn = res.get("memoryArn", "")
    return memory_arn


# =====================================================================
# TASK 6: Observability (CloudWatch & X-Ray)
# =====================================================================
def configure_observability(runtime_arn: str) -> None:
    """Enables CloudWatch logging and full 100% X-Ray distributed tracing."""
    logging_config = {
        "cloudWatchConfig": {
            "logGroup": config.AGENT_LOG_GROUP,
            "logLevel": "INFO",
            "enabled": True
        },
        "xRayConfig": {
            "enabled": True,
            "samplingRate": 1.0
        }
    }
    try:
        from src.agent_observability import apply_observability_config
        apply_observability_config(runtime_arn, logging_config)
    except Exception as e:
        logger.warning(f"Could not apply observability config: {e}")
