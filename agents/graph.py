import os
import json
from typing import TypedDict, List, Dict, Annotated
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage  # fixed: dot not underscore
from rag.retriever import CitedRetriever
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ── State Schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    company_name: str
    query: str
    retrieved_context: str
    citations: List[Dict]
    analysis: str
    calculations: str
    final_answer: str
    needs_calculation: bool


# ── LLM Setup ─────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_output_tokens=4096,
    )


# ── Node: Retriever ───────────────────────────────────────────────────────────

def retriever_node(state: AgentState) -> dict:
    """Retrieves relevant context from vector store."""
    retriever = CitedRetriever()
    context, citations = retriever.retrieve(
        query=state["query"],
        company_name=state["company_name"],
        top_k=8,
    )
    return {
        "retrieved_context": context,
        "citations": citations,
    }


# ── Node: Router ──────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Decides if financial calculation is needed."""
    CALC_KEYWORDS = [
        "calculate", "ratio", "pe", "p/e", "roe", "roce", "ebitda",
        "margin", "growth", "cagr", "eps", "debt to equity", "current ratio",
        "net profit", "revenue growth", "compare", "vs", "versus", "percentage",
    ]
    query_lower = state["query"].lower()
    needs_calc = any(kw in query_lower for kw in CALC_KEYWORDS)
    return {"needs_calculation": needs_calc}


# ── Node: Analyzer Agent ──────────────────────────────────────────────────────

def analyzer_node(state: AgentState) -> dict:
    """Qualitative analysis agent — reads context and reasons about business."""
    llm = get_llm(temperature=0.1)

    system_prompt = """You are a Senior Equity Research Analyst specializing in Indian markets.

Your job:
1. Analyze the provided source documents ONLY — never hallucinate or use outside knowledge for specific numbers
2. Answer questions based strictly on what is in the [SOURCE N] blocks
3. Reference sources by their [SOURCE N] tag inline so citations can be tracked
4. Be precise, use exact numbers from documents
5. Highlight risks and opportunities like a CFA-certified analyst would
6. If information is NOT in the sources, say "This information is not available in the provided documents"

Style: Professional equity research report tone. Structured with clear sections."""

    user_message = f"""Company: {state['company_name']}
Question: {state['query']}

Retrieved Document Sources:
{state['retrieved_context']}

Provide a detailed analysis. Reference [SOURCE N] tags whenever you use information from a source.
Structure your response with: Key Findings, Detailed Analysis, and Risks/Concerns."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    return {"analysis": response.content}


# ── Node: Calculator Agent ────────────────────────────────────────────────────

def calculator_node(state: AgentState) -> dict:
    """Financial calculator agent — extracts numbers and computes ratios."""
    if not state.get("needs_calculation"):
        return {"calculations": ""}

    llm = get_llm(temperature=0.0)

    system_prompt = """You are a Financial Calculator Agent for equity research.

Your job:
1. Extract numerical data from the provided sources
2. Calculate requested financial ratios and metrics
3. Show your formula and working clearly
4. ONLY use numbers present in the source documents
5. Format numbers clearly (₹ Crores, %, x multiples)

Common formulas you know:
- P/E Ratio = Market Price / EPS
- ROE = Net Profit / Shareholders Equity x 100
- ROCE = EBIT / Capital Employed x 100
- Debt-to-Equity = Total Debt / Shareholders Equity
- Current Ratio = Current Assets / Current Liabilities
- EBITDA Margin = EBITDA / Revenue x 100
- Revenue CAGR = (Ending Value / Beginning Value)^(1/n) - 1"""

    user_message = f"""Company: {state['company_name']}
Question requiring calculation: {state['query']}

Source Data:
{state['retrieved_context']}

Extract relevant numbers and compute the requested metrics. Show formula + working."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    return {"calculations": response.content}


# ── Node: Citation Formatter ──────────────────────────────────────────────────

def citation_node(state: AgentState) -> dict:
    """Combines analysis + calculations into final cited answer."""
    llm = get_llm(temperature=0.0)

    calc_section = ""
    if state.get("calculations"):
        calc_section = f"\n\nCalculations Section:\n{state['calculations']}"

    system_prompt = """You are a Citation Formatter for an equity research platform.

Combine the analysis and calculations into a clean, professional final answer.

Rules:
1. Keep all [SOURCE N] references inline exactly as they appear
2. Add a "## Sources Used" section at the end listing which sources were referenced
3. If calculations exist, include them in a "## Financial Calculations" section
4. Format the answer in clean markdown
5. Never add information not in the analysis"""

    citations_json = json.dumps([{
        "id": c["id"],
        "file": c["source_file"],
        "page": c["page_number"],
        "section": c["section_title"],
        "type": c["doc_type"],
    } for c in state["citations"]], indent=2)

    user_message = f"""Analysis:
{state['analysis']}
{calc_section}

Citations available:
{citations_json}

Produce the final formatted answer with citations."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    return {"final_answer": response.content}


# ── Conditional Edge ──────────────────────────────────────────────────────────

def should_calculate(state: AgentState) -> str:
    return "calculator" if state.get("needs_calculation") else "citation"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("router", router_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("citation", citation_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "router")
    graph.add_edge("router", "analyzer")
    graph.add_conditional_edges(
        "analyzer",
        should_calculate,
        {"calculator": "calculator", "citation": "citation"},
    )
    graph.add_edge("calculator", "citation")
    graph.add_edge("citation", END)

    return graph.compile()


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_query(query: str, company_name: str) -> Dict:
    """Run a query through the full agent pipeline."""
    graph = build_agent_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "company_name": company_name,
        "query": query,
        "retrieved_context": "",
        "citations": [],
        "analysis": "",
        "calculations": "",
        "final_answer": "",
        "needs_calculation": False,
    }

    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state["final_answer"],
        "citations": final_state["citations"],
        "used_calculator": final_state["needs_calculation"],
    }