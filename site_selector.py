import os
from typing import TypedDict, List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.graph import StateGraph, START, END

llm = ChatOllama(model="llama3.2", temperature=0.1, base_url="http://localhost:11434")

class SiteSelectionState(TypedDict):
    country: str
    locations: List[str]
    raw_data: Dict[str, Any]
    evaluations: Dict[str, Any]
    comparison_report: str
    report_filepath: str

def supervisor_agent_node(state: SiteSelectionState):
    country = state["country"]
    locations = state["locations"]
    
    # Initialize Serper inside the node so environment variables are fully loaded
    serper = GoogleSerperAPIWrapper()
    
    print(f"\n[SUPERVISOR] Initiating multi-agent site selection analysis for {country} across locations: {locations}")
    
    raw_data = {}
    for loc in locations:
        print(f"\n[SUPERVISOR -> WORKERS] Delegating research tasks for location: {loc}")
        
        zoning_res = serper.run(f"commercial real estate zoning environmental regulations data center {loc} {country}")
        power_res = serper.run(f"power grid capacity electricity substation renewable energy data center {loc} {country}")
        tax_res = serper.run(f"tax incentives commercial real estate data center tax abatement {loc} {country}")
        
        raw_data[loc] = {
            "zoning": zoning_res,
            "power": power_res,
            "financials": tax_res
        }
        
    return {"raw_data": raw_data}

def evaluation_subgraph_node(state: SiteSelectionState):
    raw_data = state["raw_data"]
    evaluations = {}
    
    for loc, data in raw_data.items():
        print(f"[EVALUATION AGENT] Analyzing and synthesizing data for {loc}...")
        prompt = f"""
        You are an expert Commercial Real Estate and Data Center Infrastructure Analyst.
        Analyze the following raw data for location: {loc} in {state['country']}.
        
        --- ZONING & ENVIRONMENT ---
        {data['zoning']}
        
        --- POWER & INFRASTRUCTURE ---
        {data['power']}
        
        --- FINANCIALS & INCENTIVES ---
        {data['financials']}
        
        Provide a structured assessment containing:
        1. Power Availability & Reliability (Score out of 10)
        2. Zoning & Environmental Suitability (Score out of 10)
        3. Financial Incentives & Tax Abatements (Score out of 10)
        4. Key Pros (Bulleted list)
        5. Key Cons / Risks (Bulleted list)
        """
        response = llm.invoke(prompt)
        evaluations[loc] = response.content
        
    return {"evaluations": evaluations}

def synthesis_and_report_node(state: SiteSelectionState):
    country = state["country"]
    locations = state["locations"]
    evaluations = state["evaluations"]
    
    print(f"[SUPERVISOR] Synthesizing cross-domain evaluations and generating comparison metrics...")
    eval_text = "\n\n".join([f"=== LOCATION: {loc} ===\n{eval}" for loc, eval in evaluations.items()])
    
    synthesis_prompt = f"""
    You are the Lead Investment Director. Draft a clean, crisp, and professional Commercial Real Estate & Data Center Site Selection Report comparing the following locations in {country}: {', '.join(locations)}.
    
    Here are the expert sub-agent evaluations:
    {eval_text}
    
    Your report must include:
    - Executive Summary
    - Comparative Metrics Table (ranking Power, Zoning, and Financials across locations)
    - Detailed Location Analysis with explicit **Pros** and **Cons** for each site
    - Final Recommendation / Investment Memo conclusion
    """
    
    report_content = llm.invoke(synthesis_prompt).content
    
    reports_dir = r"E:\Products\mango-lassy-chat\reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    filename_slug = "_vs_".join([loc.lower().replace(" ", "_") for loc in locations])
    filename = f"site_selection_{country.lower()}_{filename_slug}.md"
    filepath = os.path.join(reports_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"[REPORT SAVED] Successfully generated and saved report to: {filepath}")
    
    return {
        "comparison_report": report_content,
        "report_filepath": filepath
    }

# Build Site Selection Workflow
site_workflow = StateGraph(SiteSelectionState)
site_workflow.add_node("supervisor", supervisor_agent_node)
site_workflow.add_node("evaluator", evaluation_subgraph_node)
site_workflow.add_node("synthesizer", synthesis_and_report_node)

site_workflow.add_edge(START, "supervisor")
site_workflow.add_edge("supervisor", "evaluator")
site_workflow.add_edge("evaluator", "synthesizer")
site_workflow.add_edge("synthesizer", END)

site_selection_graph = site_workflow.compile()