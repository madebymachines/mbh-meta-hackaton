from openai import OpenAI
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import html
import plotly.io as pio
import json
import re
import sqlalchemy as sql
import asyncio

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict
from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
)
from ai_data_science_team.agents import SQLDatabaseAgent

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]
TITLE = "🚀 Ultimate Data Science AI Copilot"

DB_OPTIONS = {
    "None - Upload Files Only": None,
    "Northwind Database": "sqlite:///data/northwind.db",
    "Custom SQLite DB": "custom_sqlite",
    "Custom Connection String": "custom_connection"
}

st.set_page_config(
    page_title=TITLE,
    page_icon="🚀",
    layout="wide"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def render_report_iframe(report_src, src_type="url", height=620, title="Interactive Report"):
    """Render a report iframe with expandable fullscreen functionality."""
    if src_type == "html":
        iframe_src = f'srcdoc="{html.escape(report_src, quote=True)}"'
    else:
        iframe_src = f'src="{report_src}"'

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                height: 100%;
            }}
            #iframe-container {{
                position: relative;
                width: 100%;
                height: {height}px;
            }}
            #myIframe {{
                width: 100%;
                height: 100%;
                border: none;
            }}
            #fullscreen-btn {{
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
                padding: 8px 12px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div id="iframe-container">
            <button id="fullscreen-btn" onclick="toggleFullscreen()">Full Screen</button>
            <iframe id="myIframe" {iframe_src} allowfullscreen></iframe>
        </div>
        <script>
            function toggleFullscreen() {{
                var container = document.getElementById("iframe-container");
                if (!document.fullscreenElement) {{
                    container.requestFullscreen().catch(err => {{
                        alert("Error attempting to enable full-screen mode: " + err.message);
                    }});
                    document.getElementById("fullscreen-btn").innerText = "Exit Full Screen";
                }} else {{
                    document.exitFullscreen();
                    document.getElementById("fullscreen-btn").innerText = "Full Screen";
                }}
            }}
            document.addEventListener('fullscreenchange', () => {{
                if (!document.fullscreenElement) {{
                    document.getElementById("fullscreen-btn").innerText = "Full Screen";
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=True)

def setup_openai_api():
    """Setup OpenAI API and return LLM instance"""
    st.sidebar.header("🔑 OpenAI Configuration")
    
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Your OpenAI API key is required for the app to function.",
        key="openai_api_key"
    )
    
    if api_key:
        client = OpenAI(api_key=api_key)
        try:
            models = client.models.list()
            st.sidebar.success("✅ API Key is valid!")
            
            model_option = st.sidebar.selectbox(
                "Choose OpenAI model", 
                MODEL_LIST, 
                index=0,
                key="model_selection"
            )
            
            llm = ChatOpenAI(model=model_option, api_key=api_key)
            return llm, api_key
        except Exception as e:
            st.sidebar.error(f"❌ Invalid API Key: {e}")
            return None, None
    else:
        st.sidebar.info("Please enter your OpenAI API Key to proceed.")
        return None, None

def setup_database_connection():
    st.sidebar.header("🗃️ Database Configuration", divider=True)
    
    db_option = st.sidebar.selectbox(
        "Select Database Source",
        list(DB_OPTIONS.keys()),
        key="db_selection"
    )
    
    db_path = DB_OPTIONS.get(db_option)
    sql_connection = None
    
    if db_option == "None - Upload Files Only":
        st.sidebar.info("📁 File upload mode - no database connection")
        return None, None
    
    elif db_option == "Custom SQLite DB":
        uploaded_db = st.sidebar.file_uploader(
            "Upload SQLite Database (.db file)",
            type=["db", "sqlite", "sqlite3"],
            key="sqlite_upload"
        )
        if uploaded_db:
            temp_path = f"temp_{uploaded_db.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_db.read())
            try:
                sql_engine = sql.create_engine(f"sqlite:///{temp_path}")
                sql_connection = sql_engine.connect()
                st.sidebar.success(f"✅ Connected to {uploaded_db.name}")
                return sql_connection, temp_path
            except Exception as e:
                st.sidebar.error(f"❌ Failed to connect: {e}")
                return None, None
        else:
            st.sidebar.info("Please upload a SQLite database file")
            return None, None
    
    elif db_option == "Custom Connection String":
        connection_string = st.sidebar.text_input(
            "Enter Database Connection String",
            placeholder="sqlite:///path/to/db.db or postgresql://user:pass@host:port/db",
            key="custom_connection"
        )
        if connection_string:
            try:
                sql_engine = sql.create_engine(connection_string)
                sql_connection = sql_engine.connect()
                st.sidebar.success("✅ Connected to custom database")
                return sql_connection, connection_string
            except Exception as e:
                st.sidebar.error(f"❌ Failed to connect: {e}")
                return None, None
        else:
            st.sidebar.info("Please enter a connection string")
            return None, None
    
    else:
        if db_path:
            try:
                sql_engine = sql.create_engine(db_path)
                sql_connection = sql_engine.connect()
                st.sidebar.success(f"✅ Connected to {db_option}")
                return sql_connection, db_path
            except Exception as e:
                st.sidebar.error(f"❌ Failed to connect to {db_option}: {e}")
                return None, None
    
    return None, None

def setup_data_upload():
    """Handle data upload and return DataFrame"""
    st.sidebar.header("📊 File Upload (Optional)", divider=True)
    st.sidebar.markdown("*Upload files for EDA and Pandas analysis*")
    
    # Demo data option
    use_demo_data = st.sidebar.checkbox("Use demo data", value=False, key="use_demo")
    
    if use_demo_data:
        demo_file_path = Path("data/churn_data.csv")
        if demo_file_path.exists():
            df = pd.read_csv(demo_file_path)
            st.sidebar.success("✅ Demo data loaded successfully!")
            return df, "churn_data"
        else:
            st.sidebar.warning("⚠️ Demo data file not found.")
            return None, None
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file", 
        type=["csv", "xlsx", "xls"],
        key="file_upload"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            file_name = Path(uploaded_file.name).stem
            st.sidebar.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
            return df, file_name
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {e}")
            return None, None
    
    return None, None

# =============================================================================
# ROUTING SYSTEM (3-AGENT)
# =============================================================================

def detect_query_intent_llm(question: str, llm, has_database: bool = False) -> str:
    """
    Use LLM to intelligently determine which agent to use based on user intent
    Returns: 'eda', 'pandas', or 'sql'
    """
    
    agents_description = """
**Available Agents:**

1. **EDA Agent** 🔍 - For comprehensive data exploration and analysis:
   - Dataset descriptions, summaries, and overviews
   - Missing data analysis and data quality checks
   - Correlation analysis and statistical relationships
   - Data profiling and comprehensive reports (Sweetviz, D-Tale)
   - Distribution analysis and outlier detection
   - Data validation and integrity checks

2. **Pandas Agent** 🐼 - For specific data queries and visualizations:
   - Filtering, sorting, and data manipulation on uploaded files
   - Creating specific charts and visualizations
   - Answering business questions with uploaded data
   - Top/bottom N queries and rankings
   - Aggregations (sum, count, average, etc.)
   - Comparisons and trend analysis
"""

    if has_database:
        response_format = "Return ONLY one word: either \"eda\", \"pandas\", or \"sql\""
    else:
        response_format = "Return ONLY one word: either \"eda\" or \"pandas\""
    
    routing_prompt = f"""
You are a smart routing system for a data science application. Based on the user's question, you need to determine which AI agent should handle the query.

{agents_description}

**User Question:** "{question}"

**Database Available:** {"Yes" if has_database else "No"}

**Instructions:**
- Analyze the user's intent and context carefully
- If the question is about understanding data structure/quality → choose EDA
- If the question is about getting specific insights from uploaded files → choose Pandas
{"- If the question mentions SQL, tables, database queries, or complex joins → choose SQL" if has_database else ""}
{"- If the question could benefit from database operations → choose SQL" if has_database else ""}
- For ambiguous cases, consider which agent would provide the most value

**Response Format:**
{response_format}
"""
    
    try:
        response = llm.invoke(routing_prompt)
        decision = response.content.strip().lower()
        
        # Validate response
        valid_options = ['eda', 'pandas', 'sql'] if has_database else ['eda', 'pandas']
        if decision in valid_options:
            return decision
        else:
            # Fallback to keyword-based if LLM gives unexpected response
            return detect_query_intent_fallback(question, has_database)
    except Exception as e:
        # Fallback to keyword-based routing if LLM fails
        print(f"LLM routing failed: {e}")
        return detect_query_intent_fallback(question, has_database)

def detect_query_intent_fallback(question: str, has_database: bool = False) -> str:
    """
    Fallback keyword-based routing for when LLM routing fails
    """
    question_lower = question.lower()
    
    if has_database:
        sql_indicators = [
            'sql', 'query', 'select', 'from', 'where', 'join', 'group by',
            'table', 'database', 'db', 'schema', 'tables', 'columns',
            'insert', 'update', 'delete', 'create', 'alter', 'drop',
            'aggregat', 'sum', 'count', 'avg', 'max', 'min',
            'inner join', 'left join', 'right join', 'full join'
        ]
        
        sql_score = sum(1 for indicator in sql_indicators if indicator in question_lower)

        if any(strong_sql in question_lower for strong_sql in ['sql', 'query', 'database', 'table', 'join']):
            return 'sql'
    else:
        sql_score = 0
    
    # EDA indicators
    eda_indicators = [
        'describe', 'summary', 'overview', 'analyze', 'analysis',
        'missing', 'null', 'quality', 'profile', 'profiling',
        'correlation', 'relationship', 'distribution',
        'sweetviz', 'dtale', 'report', 'comprehensive',
        'statistics', 'statistical', 'outlier', 'anomaly',
        'explore', 'exploration', 'exploratory'
    ]
    
    # Pandas indicators
    pandas_indicators = [
        'show', 'display', 'plot', 'chart', 'graph', 'visualize',
        'top', 'bottom', 'highest', 'lowest', 'best', 'worst',
        'filter', 'sort', 'group', 'by month', 'by year',
        'average', 'compare', 'trend', 'bar chart', 'pie chart'
    ]
    
    eda_score = sum(1 for indicator in eda_indicators if indicator in question_lower)
    pandas_score = sum(1 for indicator in pandas_indicators if indicator in question_lower)
    
    if has_database and sql_score > max(eda_score, pandas_score):
        return 'sql'
    elif eda_score > pandas_score:
        return 'eda'
    else:
        return 'pandas'

def detect_query_intent(question: str, llm=None, use_llm=True, has_database=False) -> str:
    """
    Main routing function - uses LLM by default, falls back to keywords
    """
    if use_llm and llm:
        return detect_query_intent_llm(question, llm, has_database)
    else:
        return detect_query_intent_fallback(question, has_database)

# =============================================================================
# EDA FUNCTIONS
# =============================================================================

def process_exploratory(question: str, llm, data: pd.DataFrame) -> dict:
    """Process EDA queries using EDAToolsAgent"""
    eda_agent = EDAToolsAgent(
        llm,
        invoke_react_agent_kwargs={"recursion_limit": 10},
    )

    question += " Don't return hyperlinks to files in the response."

    eda_agent.invoke_agent(
        user_instructions=question,
        data_raw=data,
    )

    tool_calls = eda_agent.get_tool_calls()
    ai_message = eda_agent.get_ai_message(markdown=False)
    artifacts = eda_agent.get_artifacts(as_dataframe=False)

    result = {
        "ai_message": ai_message,
        "tool_calls": tool_calls,
        "artifacts": artifacts,
        "agent_type": "eda"
    }

    if tool_calls:
        last_tool_call = tool_calls[-1]
        result["last_tool_call"] = last_tool_call
        tool_name = last_tool_call

        if tool_name == "describe_dataset":
            if artifacts and isinstance(artifacts, dict) and "describe_df" in artifacts:
                try:
                    df = pd.DataFrame(artifacts["describe_df"])
                    result["describe_df"] = df
                except Exception as e:
                    st.error(f"Error processing describe_dataset artifact: {e}")

        elif tool_name == "visualize_missing":
            if artifacts and isinstance(artifacts, dict):
                try:
                    matrix_fig = matplotlib_from_base64(artifacts.get("matrix_plot"))
                    bar_fig = matplotlib_from_base64(artifacts.get("bar_plot"))
                    heatmap_fig = matplotlib_from_base64(artifacts.get("heatmap_plot"))
                    result["matrix_plot_fig"] = matrix_fig[0]
                    result["bar_plot_fig"] = bar_fig[0]
                    result["heatmap_plot_fig"] = heatmap_fig[0]
                except Exception as e:
                    st.error(f"Error processing visualize_missing artifact: {e}")

        elif tool_name == "generate_correlation_funnel":
            if artifacts and isinstance(artifacts, dict):
                if "correlation_data" in artifacts:
                    try:
                        corr_df = pd.DataFrame(artifacts["correlation_data"])
                        result["correlation_data"] = corr_df
                    except Exception as e:
                        st.error(f"Error processing correlation_data: {e}")
                if "plotly_figure" in artifacts:
                    try:
                        corr_plotly = plotly_from_dict(artifacts["plotly_figure"])
                        result["correlation_plotly"] = corr_plotly
                    except Exception as e:
                        st.error(f"Error processing correlation funnel Plotly figure: {e}")

        elif tool_name == "generate_sweetviz_report":
            if artifacts and isinstance(artifacts, dict):
                result["report_file"] = artifacts.get("report_file")
                result["report_html"] = artifacts.get("report_html")

        elif tool_name == "generate_dtale_report":
            if artifacts and isinstance(artifacts, dict):
                result["dtale_url"] = artifacts.get("dtale_url")

        else:
            if artifacts and isinstance(artifacts, dict):
                if "plotly_figure" in artifacts:
                    try:
                        plotly_fig = plotly_from_dict(artifacts["plotly_figure"])
                        result["plotly_fig"] = plotly_fig
                    except Exception as e:
                        st.error(f"Error processing Plotly figure: {e}")
                if "plot_image" in artifacts:
                    try:
                        fig = matplotlib_from_base64(artifacts["plot_image"])
                        result["matplotlib_fig"] = fig
                    except Exception as e:
                        st.error(f"Error processing matplotlib image: {e}")
                if "dataframe" in artifacts:
                    try:
                        df = pd.DataFrame(artifacts["dataframe"])
                        result["dataframe"] = df
                    except Exception as e:
                        st.error(f"Error converting artifact to dataframe: {e}")

    return result

# =============================================================================
# PANDAS ANALYST FUNCTIONS
# =============================================================================

def process_pandas_query(question: str, llm, data: pd.DataFrame, pandas_analyst) -> dict:
    """Process Pandas queries using PandasDataAnalyst"""
    try:
        pandas_analyst.invoke_agent(
            user_instructions=question,
            data_raw=data,
        )
        result = pandas_analyst.get_response()
        result["agent_type"] = "pandas"
        result["success"] = True
        return result
    except Exception as e:
        return {
            "agent_type": "pandas",
            "success": False,
            "error": str(e),
            "ai_message": "An error occurred while processing your pandas query. Please try again."
        }

# =============================================================================
# SQL FUNCTIONS
# =============================================================================

async def process_sql_query(question: str, llm, sql_connection) -> dict:
    """Process SQL queries using SQLDatabaseAgent"""
    try:
        sql_db_agent = SQLDatabaseAgent(
            model=llm,
            connection=sql_connection,
            n_samples=1,
            log=False,
            bypass_recommended_steps=True,
        )
        
        await sql_db_agent.ainvoke_agent(user_instructions=question)
        
        sql_query = sql_db_agent.get_sql_query_code()
        response_df = sql_db_agent.get_data_sql()
        
        return {
            "agent_type": "sql",
            "success": True,
            "sql_query": sql_query,
            "dataframe": response_df,
            "ai_message": f"Successfully executed SQL query."
        }
        
    except Exception as e:
        return {
            "agent_type": "sql",
            "success": False,
            "error": str(e),
            "ai_message": f"An error occurred while processing your SQL query: {str(e)}"
        }

# =============================================================================
# UNIFIED CHAT DISPLAY
# =============================================================================

def display_chat_history():
    """Display unified chat history with artifacts from all agents"""
    for i, msg in enumerate(st.session_state.unified_msgs.messages):
        with st.chat_message(msg.type):
            if msg.content.startswith("AGENT_RESPONSE:"):
                continue
            elif "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                if plot_index < len(st.session_state.plots):
                    st.plotly_chart(
                        st.session_state.plots[plot_index], 
                        key=f"plot_{plot_index}_{i}",
                        use_container_width=True
                    )
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                if df_index < len(st.session_state.dataframes):
                    st.dataframe(
                        st.session_state.dataframes[df_index],
                        key=f"df_{df_index}_{i}",
                        use_container_width=True
                    )
            elif "SQL_RESULT:" in msg.content:
                content = msg.content.replace("SQL_RESULT:", "").strip()
                st.markdown(content)
            else:
                st.write(msg.content)
            
            # Display EDA artifacts
            if (
                "chat_artifacts" in st.session_state
                and i in st.session_state["chat_artifacts"]
            ):
                for artifact in st.session_state["chat_artifacts"][i]:
                    with st.expander(artifact["title"], expanded=True):
                        if artifact["render_type"] == "dataframe":
                            st.dataframe(artifact["data"], use_container_width=True)
                        elif artifact["render_type"] == "matplotlib":
                            st.pyplot(artifact["data"])
                        elif artifact["render_type"] == "plotly":
                            st.plotly_chart(artifact["data"], use_container_width=True)
                        elif artifact["render_type"] == "sweetviz":
                            report_file = artifact["data"].get("report_file")
                            try:
                                with open(report_file, "r", encoding="utf-8") as f:
                                    report_html = f.read()
                            except Exception as e:
                                st.error(f"Could not open report file: {e}")
                                report_html = "<h1>Report not found</h1>"

                            render_report_iframe(
                                report_html,
                                src_type="html",
                                height=620,
                                title="Sweetviz Report",
                            )
                        elif artifact["render_type"] == "dtale":
                            dtale_url = artifact["data"]["dtale_url"]
                            render_report_iframe(
                                dtale_url,
                                src_type="url",
                                height=620,
                                title="D-Tale Report",
                            )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.title(TITLE)
    
    st.markdown("""
    🎯 **The Ultimate Data Science Copilot!** 
    
    Ask any question, we have:
    
    - 🔍 **EDA Agent** for comprehensive data exploration and reports
    - 🐼 **Pandas Agent** for file-based analysis and visualizations  
    - 🗃️ **SQL Agent** for database queries and complex operations
    """)
    
    # Setup API and connections
    llm, api_key = setup_openai_api()
    if not llm:
        st.stop()
    
    sql_connection, db_path = setup_database_connection()
    data, file_name = setup_data_upload()
    
    # Check what data sources are available
    has_database = sql_connection is not None
    has_files = data is not None
    
    if not has_database and not has_files:
        st.warning("⚠️ Please either connect to a database or upload a file to proceed.")
        st.stop()
    
    # Display data sources status
    col1, col2 = st.columns(2)
    with col1:
        if has_database:
            st.success(f"🗃️ **Database Connected:** {db_path or 'Custom Connection'}")
        else:
            st.info("🗃️ **Database:** Not connected")
    
    with col2:
        if has_files:
            st.success(f"📁 **File Loaded:** {file_name}")
        else:
            st.info("📁 **Files:** None uploaded")
    
    # Display data preview for files
    if data is not None:
        with st.expander(f"📋 File Preview: {file_name}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(data.head(), use_container_width=True)
            with col2:
                st.metric("Rows", data.shape[0])
                st.metric("Columns", data.shape[1])
    
    # Initialize session state
    if "unified_msgs" not in st.session_state:
        st.session_state.unified_msgs = StreamlitChatMessageHistory(key="unified_messages")
        if len(st.session_state.unified_msgs.messages) == 0:
            welcome_msg = "Hi! 👋 I'm your Ultimate Data Science Copilot with 3 specialized AI agents:\n\n"
            if has_database:
                welcome_msg += "🗃️ **SQL Agent** - Ask about database tables, run queries\n"
            if has_files:
                welcome_msg += "🔍 **EDA Agent** - Explore and analyze your uploaded data\n🐼 **Pandas Agent** - Create charts and get insights from files\n"
            welcome_msg += "\n**Try asking:**\n"
            if has_database:
                welcome_msg += "- 'What tables exist in the database?' (SQL)\n"
            if has_files:
                welcome_msg += "- 'Describe this dataset' (EDA)\n- 'Show top 5 records in a bar chart' (Pandas)\n"
            
            st.session_state.unified_msgs.add_ai_message(welcome_msg)
    
    if "chat_artifacts" not in st.session_state:
        st.session_state["chat_artifacts"] = {}
    
    if "plots" not in st.session_state:
        st.session_state.plots = []
    
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = []
    
    # Setup Pandas Data Analyst (initialize once)
    if "pandas_analyst" not in st.session_state and has_files:
        st.session_state.pandas_analyst = PandasDataAnalyst(
            model=llm,
            data_wrangling_agent=DataWranglingAgent(
                model=llm,
                log=False,
                bypass_recommended_steps=True,
                n_samples=100,
            ),
            data_visualization_agent=DataVisualizationAgent(
                model=llm,
                n_samples=100,
                log=False,
            ),
        )
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if question := st.chat_input("Ask me anything about your data...", key="unified_input"):
        with st.spinner("🤖 Analyzing your question and choosing the best AI agent..."):
            
            # Add user message
            st.session_state.unified_msgs.add_user_message(question)
            
            # Detect intent and route to appropriate agent
            intent = detect_query_intent(question, llm, use_llm_routing, has_database)
            
            # Validate intent based on available resources
            if intent == 'sql' and not has_database:
                intent = 'pandas' if has_files else 'eda'
            elif intent in ['eda', 'pandas'] and not has_files:
                if has_database:
                    intent = 'sql'
                else:
                    st.error("No data source available for this query.")
                    return
            
            # Route to appropriate agent
            if intent == 'sql':
                st.info("🗃️ **SQL Agent** selected for database operations")
                
                async def handle_sql():
                    return await process_sql_query(question, llm, sql_connection)
                
                try:
                    result = asyncio.run(handle_sql())
                    
                    if result.get("success", False):
                        sql_query = result.get("sql_query", "")
                        response_df = result.get("dataframe")
                        
                        if sql_query:
                            # Format response with SQL query
                            response_text = f"### 🗃️ SQL Query Results\n\n**SQL Query:**\n```sql\n{sql_query}\n```\n\n**Result:**"
                            
                            # Store dataframe
                            df_index = len(st.session_state.dataframes)
                            st.session_state.dataframes.append(response_df)
                            
                            # Add messages
                            st.session_state.unified_msgs.add_ai_message(response_text)
                            st.session_state.unified_msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                            
                            # Display results
                            st.chat_message("ai").write(response_text)
                            st.dataframe(response_df, use_container_width=True)
                        else:
                            error_msg = "❌ No SQL query was generated."
                            st.session_state.unified_msgs.add_ai_message(error_msg)
                            st.chat_message("ai").write(error_msg)
                    else:
                        error_msg = result.get("ai_message", "An error occurred with the SQL query.")
                        st.session_state.unified_msgs.add_ai_message(error_msg)
                        st.chat_message("ai").write(error_msg)
                        
                except Exception as e:
                    error_msg = f"❌ SQL Agent error: {str(e)}"
                    st.session_state.unified_msgs.add_ai_message(error_msg)
                    st.chat_message("ai").write(error_msg)
            
            elif intent == 'eda':
                st.info("🔍 **EDA Agent** selected for comprehensive data analysis")
                result = process_exploratory(question, llm, data)
                
                # Process EDA response
                ai_msg = result.get("ai_message", "")
                tool_name = result.get("last_tool_call", "")
                if tool_name:
                    ai_msg += f"\n\n*🔧 Tool Used: {tool_name}*"
                
                st.session_state.unified_msgs.add_ai_message(ai_msg)
                
                # Process EDA artifacts
                artifact_list = []
                if "last_tool_call" in result:
                    tool_name = result["last_tool_call"]
                    if tool_name == "describe_dataset" and "describe_df" in result:
                        artifact_list.append({
                            "title": "📊 Dataset Description",
                            "render_type": "dataframe",
                            "data": result["describe_df"],
                        })
                    elif tool_name == "visualize_missing":
                        if "matrix_plot_fig" in result:
                            artifact_list.append({
                                "title": "🔍 Missing Data Matrix",
                                "render_type": "matplotlib",
                                "data": result["matrix_plot_fig"],
                            })
                        if "bar_plot_fig" in result:
                            artifact_list.append({
                                "title": "📊 Missing Data Bar Plot",
                                "render_type": "matplotlib",
                                "data": result["bar_plot_fig"],
                            })
                        if "heatmap_plot_fig" in result:
                            artifact_list.append({
                                "title": "🔥 Missing Data Heatmap",
                                "render_type": "matplotlib",
                                "data": result["heatmap_plot_fig"],
                            })
                    elif tool_name == "generate_correlation_funnel":
                        if "correlation_data" in result:
                            artifact_list.append({
                                "title": "📈 Correlation Data",
                                "render_type": "dataframe",
                                "data": result["correlation_data"],
                            })
                        if "correlation_plotly" in result:
                            artifact_list.append({
                                "title": "🎯 Interactive Correlation Funnel",
                                "render_type": "plotly",
                                "data": result["correlation_plotly"],
                            })
                    elif tool_name == "generate_sweetviz_report":
                        artifact_list.append({
                            "title": "📋 Sweetviz Comprehensive Report",
                            "render_type": "sweetviz",
                            "data": {
                                "report_file": result.get("report_file"),
                                "report_html": result.get("report_html"),
                            },
                        })
                    elif tool_name == "generate_dtale_report":
                        artifact_list.append({
                            "title": "🔗 D-Tale Interactive Report",
                            "render_type": "dtale",
                            "data": {"dtale_url": result.get("dtale_url")},
                        })
                    else:
                        # Handle other EDA artifacts
                        if "plotly_fig" in result:
                            artifact_list.append({
                                "title": "📊 Interactive Visualization",
                                "render_type": "plotly",
                                "data": result["plotly_fig"],
                            })
                        if "matplotlib_fig" in result:
                            artifact_list.append({
                                "title": "📈 Chart",
                                "render_type": "matplotlib",
                                "data": result["matplotlib_fig"],
                            })
                        if "dataframe" in result:
                            artifact_list.append({
                                "title": "📋 Data Result",
                                "render_type": "dataframe",
                                "data": result["dataframe"],
                            })
                
                if artifact_list:
                    msg_index = len(st.session_state.unified_msgs.messages) - 1
                    st.session_state["chat_artifacts"][msg_index] = artifact_list
            
            else:  # pandas agent
                st.info("🐼 **Pandas Agent** selected for data analysis and visualization")
                result = process_pandas_query(question, llm, data, st.session_state.pandas_analyst)
                
                if result.get("success", False):
                    routing = result.get("routing_preprocessor_decision")
                    
                    if routing == "chart" and not result.get("plotly_error", False):
                        # Process chart result
                        plot_data = result.get("plotly_graph")
                        if plot_data:
                            if isinstance(plot_data, dict):
                                plot_json = json.dumps(plot_data)
                            else:
                                plot_json = plot_data
                            plot_obj = pio.from_json(plot_json)
                            
                            response_text = "📊 Here's your visualization:"
                            plot_index = len(st.session_state.plots)
                            st.session_state.plots.append(plot_obj)
                            
                            st.session_state.unified_msgs.add_ai_message(response_text)
                            st.session_state.unified_msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                            
                            st.chat_message("ai").write(response_text)
                            st.plotly_chart(plot_obj, use_container_width=True)
                        else:
                            error_msg = "❌ The agent could not generate a valid chart."
                            st.session_state.unified_msgs.add_ai_message(error_msg)
                            st.chat_message("ai").write(error_msg)
                    
                    elif routing == "table":
                        # Process table result
                        data_wrangled = result.get("data_wrangled")
                        if data_wrangled is not None:
                            response_text = "📋 Here's your data result:"
                            if not isinstance(data_wrangled, pd.DataFrame):
                                data_wrangled = pd.DataFrame(data_wrangled)
                            
                            df_index = len(st.session_state.dataframes)
                            st.session_state.dataframes.append(data_wrangled)
                            
                            st.session_state.unified_msgs.add_ai_message(response_text)
                            st.session_state.unified_msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                            
                            st.chat_message("ai").write(response_text)
                            st.dataframe(data_wrangled, use_container_width=True)
                        else:
                            error_msg = "❌ No data was returned by the agent."
                            st.session_state.unified_msgs.add_ai_message(error_msg)
                            st.chat_message("ai").write(error_msg)
                    else:
                        # Fallback case
                        data_wrangled = result.get("data_wrangled")
                        if data_wrangled is not None:
                            response_text = "⚠️ There was an issue generating the chart. Here's the data instead:"
                            if not isinstance(data_wrangled, pd.DataFrame):
                                data_wrangled = pd.DataFrame(data_wrangled)
                            
                            df_index = len(st.session_state.dataframes)
                            st.session_state.dataframes.append(data_wrangled)
                            
                            st.session_state.unified_msgs.add_ai_message(response_text)
                            st.session_state.unified_msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                            
                            st.chat_message("ai").write(response_text)
                            st.dataframe(data_wrangled, use_container_width=True)
                        else:
                            error_msg = "❌ An error occurred while processing your query. Please try again."
                            st.session_state.unified_msgs.add_ai_message(error_msg)
                            st.chat_message("ai").write(error_msg)
                else:
                    # Handle error case
                    error_msg = result.get("ai_message", "An error occurred while processing your query.")
                    st.session_state.unified_msgs.add_ai_message(error_msg)
                    st.chat_message("ai").write(error_msg)
            
            st.rerun()

    # Add helpful sidebar info
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Routing Settings")
        
        use_llm_routing = st.checkbox(
            "🤖 Use LLM-based Smart Routing", 
            value=True,
            help="Use AI to intelligently route queries (more accurate but slower)"
        )
        
        if not use_llm_routing:
            st.info("Using keyword-based routing (faster but less accurate)")
        
        st.markdown("---")
        st.markdown("### 🤖 Available AI Agents")
        
        if has_database:
            st.markdown("🗃️ **SQL Agent** - Database queries")
        else:
            st.markdown("🗃️ **SQL Agent** - ❌ No database connected")
            
        if has_files:
            st.markdown("🔍 **EDA Agent** - Data exploration")
            st.markdown("🐼 **Pandas Agent** - Analysis & charts")
        else:
            st.markdown("🔍 **EDA Agent** - ❌ No files uploaded")
            st.markdown("🐼 **Pandas Agent** - ❌ No files uploaded")
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        
        example_questions = []
        
        if has_database:
            example_questions.extend([
                "What tables exist in the database?",
                "Show me first 10 rows from [table_name]",
                "Aggregate sales by region using SQL"
            ])
        
        if has_files:
            example_questions.extend([
                "Describe this dataset",
                "Show top 5 records in a bar chart", 
                "Analyze missing data",
                "Create correlation report"
            ])
        
        for i, eq in enumerate(example_questions):
            if st.button(eq, key=f"example_{i}", use_container_width=True):
                st.info(f"Example: {eq}")

if __name__ == "__main__":
    main()