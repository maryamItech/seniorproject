"""Streamlit app for RAG system using LangChain and LangGraph."""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import re
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add project root to path BEFORE any other imports
# This ensures all local modules can be imported
# Get the directory where this script is located
script_dir = Path(__file__).parent
# Get project root (parent of rag/ directory)
project_root = script_dir.parent
project_root_str = str(project_root.resolve())

# Add to Python path
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Also set as environment variable for subprocesses
os.environ['PYTHONPATH'] = project_root_str + os.pathsep + os.environ.get('PYTHONPATH', '')

# Now import Streamlit
import streamlit as st

# Import local modules (after path is set)
try:
    from retrieval.multimodal_milvus_store import MultimodalMilvusStore
    from graph.rag_graph import RAGGraph
    from tools.evaluation_tool import evaluate_rag_with_mistral, format_retrieved_context_for_evaluation
except ImportError as e:
    # Check if it's a missing dependency error
    error_str = str(e)
    is_dependency_error = any(
        dep in error_str.lower() 
        for dep in ['pandas', 'numpy', 'datasets', 'faiss', 'langchain', 'streamlit']
    )
    
    # Show detailed error information
    error_msg = f"""
    **Import Error: {error_str}**
    
    **Troubleshooting:**
    - Project root: `{project_root_str}`
    - Script location: `{script_dir}`
    - Python path (first 3 entries):
      {chr(10).join(f'  - {p}' for p in sys.path[:3])}
    
    **Solution:**
    """
    
    if is_dependency_error:
        error_msg += """
    1. **Install missing dependencies:**
       ```bash
       pip install -r requirements.txt
       ```
       
       Or install specific missing package (check error message above).
    """
    else:
        error_msg += """
    1. Make sure you're running from the project root:
       ```bash
       streamlit run rag/app.py
       ```
    
    2. Verify all modules exist:
       - `retrieval/faiss_store.py`
       - `graph/rag_graph.py`
       - `chains/`
       - `config/`
       - `utils/`
    """
    
    error_msg += """
    3. Run test script to verify imports:
       ```bash
       python test_imports.py
       ```
    """
    
    st.error(error_msg)
    st.stop()


@st.cache_resource
def load_rag_system():
    """
    Load RAG system components (cached for performance).
    
    This function uses Streamlit's @st.cache_resource to ensure all expensive
    components are loaded only once and reused across all queries.
    """
    # Import component manager functions (they use @st.cache_resource internally)
    from utils.component_manager import (
        get_vector_store,
        get_correction_llm,
        get_generation_llm,
    )
    from graph.rag_graph import RAGGraph
    
    # Get cached components (loaded only once)
    vector_store = get_vector_store()
    correction_llm = get_correction_llm()
    generation_llm = get_generation_llm()
    
    # Create RAG graph with pre-initialized components
    rag_graph = RAGGraph(
        vector_store=vector_store,
        correction_llm=correction_llm,
        generation_llm=generation_llm,
    )
    
    return rag_graph


def initialize_session_state():
    """Initialize session state variables for conversation management."""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "workflow_status" not in st.session_state:
        st.session_state.workflow_status = {
            "correct_question": "pending",
            "rephrase_question": "pending",
            "routing_node": "pending",
            "image_processing_node": "pending",
            "perception_routing_node": "pending",
            "visual_understanding_node": "pending",
            "ocr_node": "pending",
            "correct_question": "pending",
            "rephrase_question": "pending",
            "split_question": "pending",
            "kb_query_node": "pending",
            "presentation_control_node": "pending",
            "generate_answer": "pending"
        }
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "resolved_question" not in st.session_state:
        st.session_state.resolved_question = ""
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = {}
    if "processing" not in st.session_state:
        st.session_state.processing = False


def detect_followup_question(new_question: str, conversation_history: List[Dict]) -> Optional[str]:
    """
    Detect if a question is a follow-up and resolve it with context.
    
    Returns the resolved question if it's a follow-up, None otherwise.
    """
    if not conversation_history:
        return None
    
    # Get the last question and answer
    last_qa = conversation_history[-1]
    last_question = last_qa.get("question", "")
    last_answer = last_qa.get("answer", "")
    
    # Strong indicators of follow-up questions
    strong_followup_indicators = [
        r'^\s*(what about|how about|tell me more|explain more|describe|elaborate|can you|could you)\b',
        r'\b(it|this|that|they|them|these|those)\b.*\?',
        r'^\s*(and|also|too|as well|additionally|furthermore|moreover)\b',
        r'\b(same|similar|related|about that|regarding|concerning)\b'
    ]
    
    # Check if the question contains strong follow-up indicators
    is_followup = any(re.search(pattern, new_question.lower()) for pattern in strong_followup_indicators)
    
    # Also check if question is very short (likely a follow-up)
    if len(new_question.split()) <= 5:
        # Check for pronouns or references
        if re.search(r'\b(it|this|that|they|them|these|those|he|she|they)\b', new_question.lower()):
            is_followup = True
    
    # Check if question starts with common question words but seems incomplete
    if not is_followup and re.match(r'^\s*(what|who|where|when|why|how|which)\s+\w+\s*\?$', new_question.lower()):
        # Very short questions are likely follow-ups
        if len(new_question.split()) <= 4:
            is_followup = True
    
    if is_followup:
        # Resolve the follow-up by incorporating context
        # Create a more natural resolved question that the RAG system can process
        context_summary = f"Based on the previous conversation where the question was '{last_question}' and the answer was '{last_answer[:200]}...', "
        resolved = f"{context_summary}please answer this follow-up question: {new_question}"
        return resolved
    
    return None


def render_workflow_graph(
    status_dict: Dict[str, str],
    image_path: Optional[str] = None,
    metadata: Optional[Dict] = None,
    container=None,
    vertical: bool = True
):
    """
    Render a visual workflow graph showing the RAG pipeline steps.
    
    Args:
        status_dict: Dictionary mapping step names to status ("pending", "active", "completed")
        image_path: Path to the image being processed (if any)
        metadata: Optional metadata dict with additional info (e.g., retrieved_docs_count, etc.)
        container: Streamlit container to render in (None means use st directly)
        vertical: If True, display vertically (for sidebar), else horizontally
    """
    steps = [
        ("routing_node", "Input Routing", "🚦", []),
        ("image_processing_node", "Image Processing", "🖼️", ["image_path"]),
        ("perception_routing_node", "Perception & Routing", "👁️", ["image_path"]),
        ("visual_understanding_node", "Image-to-Text Description", "🔍", ["image_path"]),
        ("ocr_node", "OCR Extraction", "📝", ["image_path"]),
        ("correct_question", "Correct Question", "✏️", []),
        ("rephrase_question", "Rephrase Question", "🔄", []),
        ("split_question", "Split Question", "✂️", []),
        ("kb_query_node", "Knowledge Retrieval", "🔍", []),
        ("presentation_control_node", "Select Images", "🖼️", []),
        ("generate_answer", "Generate Answer", "💬", []),
    ]
    
    # Build metadata dict if not provided
    if metadata is None:
        metadata = {}
    
    # Status colors and icons
    status_config = {
        "pending": ("⚪", "#E0E0E0", "Pending"),
        "active": ("🟡", "#FFD700", "Active"),
        "completed": ("🟢", "#4CAF50", "Completed")
    }
    
    # Use container if provided, otherwise use st directly (no with statement)
    if container is None:
        render_func = st
    else:
        render_func = container
    
    if not vertical:
        render_func.markdown("### 🔄 RAG Pipeline Workflow")
        if image_path:
            render_func.caption(f"🖼️ **Processing Image:** `{image_path}`")
    
    if vertical:
        # Vertical layout for sidebar
        for idx, (step_key, step_name, emoji, image_fields) in enumerate(steps):
            status = status_dict.get(step_key, "pending")
            icon, color, status_text = status_config.get(status, ("⚪", "#E0E0E0", "Pending"))
            
            # Build additional info text
            info_parts = []
            if image_path and "image_path" in image_fields:
                # Show truncated image path
                img_display = image_path.split("\\")[-1] if "\\" in image_path else image_path.split("/")[-1]
                info_parts.append(f"📷 {img_display[:30]}...")
            
            # Add metadata if available
            if step_key == "kb_query_node" and metadata.get("retrieved_docs_count"):
                info_parts.append(f"📚 {metadata['retrieved_docs_count']} docs")
            
            info_text = "<br>".join(info_parts) if info_parts else ""
            
            # Create a compact card for sidebar
            render_func.markdown(
                f"""
                <div style="
                    background-color: {color}20;
                    border: 2px solid {color};
                    border-radius: 8px;
                    padding: 10px;
                    text-align: center;
                    margin: 5px 0;
                ">
                    <div style="font-size: 18px; margin-bottom: 5px;">
                        {emoji} {icon}
                    </div>
                    <div style="font-weight: bold; font-size: 11px; margin-bottom: 3px;">
                        {step_name}
                    </div>
                    <div style="font-size: 10px; color: #666; margin-bottom: 3px;">
                        {status_text}
                    </div>
                    {f'<div style="font-size: 9px; color: #888; margin-top: 5px;">{info_text}</div>' if info_text else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Add arrow between steps (except for last)
            if idx < len(steps) - 1:
                render_func.markdown(
                    '<div style="text-align: center; margin: 2px 0; font-size: 16px;">↓</div>',
                    unsafe_allow_html=True
                )
    else:
        # Horizontal layout for main area - scrollable
        render_func.markdown(
            '<div style="overflow-x: auto; white-space: nowrap;">',
            unsafe_allow_html=True
        )
        
        # Create columns for steps (use wider columns for better visibility)
        num_cols = min(len(steps), 6)  # Show 6 columns at a time
        cols = render_func.columns(num_cols)
        
        for idx, (step_key, step_name, emoji, image_fields) in enumerate(steps):
            col_idx = idx % num_cols
            if idx > 0 and col_idx == 0:
                # Create new row of columns
                cols = render_func.columns(num_cols)
            
            with cols[col_idx]:
                status = status_dict.get(step_key, "pending")
                icon, color, status_text = status_config.get(status, ("⚪", "#E0E0E0", "Pending"))
                
                # Build additional info
                info_parts = []
                if image_path and "image_path" in image_fields:
                    img_display = image_path.split("\\")[-1] if "\\" in image_path else image_path.split("/")[-1]
                    info_parts.append(f"📷 {img_display[:25]}")
                
                if step_key == "kb_query_node" and metadata.get("retrieved_docs_count"):
                    info_parts.append(f"📚 {metadata['retrieved_docs_count']} docs")
                
                info_text = " | ".join(info_parts) if info_parts else ""
                
                # Create a card-like display
                render_func.markdown(
                    f"""
                    <div style="
                        background-color: {color}20;
                        border: 2px solid {color};
                        border-radius: 10px;
                        padding: 12px;
                        text-align: center;
                        margin: 5px;
                        min-height: 140px;
                        min-width: 120px;
                    ">
                        <div style="font-size: 24px; margin-bottom: 8px;">
                            {emoji} {icon}
                        </div>
                        <div style="font-weight: bold; font-size: 12px; margin-bottom: 5px; line-height: 1.2;">
                            {step_name}
                        </div>
                        <div style="font-size: 11px; color: #666; margin-bottom: 3px;">
                            {status_text}
                        </div>
                        {f'<div style="font-size: 9px; color: #888; margin-top: 5px; word-wrap: break-word;">{info_text}</div>' if info_text else ''}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        render_func.markdown('</div>', unsafe_allow_html=True)


def update_workflow_status(step: str, status: str):
    """Update the status of a workflow step."""
    if "workflow_status" in st.session_state:
        st.session_state.workflow_status[step] = status


def reset_workflow_status():
    """Reset all workflow steps to pending."""
    if "workflow_status" in st.session_state:
        for step in st.session_state.workflow_status:
            st.session_state.workflow_status[step] = "pending"


def clear_conversation():
    """Clear conversation history and reset workflow."""
    st.session_state.conversation_history = []
    st.session_state.current_query = ""
    st.session_state.resolved_question = ""
    st.session_state.pipeline_state = {}
    st.session_state.processing = False
    reset_workflow_status()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Advanced ScienceQA RAG - LangChain + LangGraph",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for modern UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .section-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .resolved-question-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"><h1>📚 ScienceQA RAG System</h1><p>Powered by LangChain + LangGraph</p></div>', unsafe_allow_html=True)
    
    # Sidebar with workflow graph
    with st.sidebar:
        st.header("⚙️ System Information")
        st.info("""
        **Architecture:**
        - LangChain for RAG pipeline
        - LangGraph for workflow orchestration
        - Milvus (Zilliz Cloud) vector store
        - BGE embeddings
        - Ollama (Qwen3-vl:4b) LLM
        """)
        
        st.markdown("---")
        st.header("🔄 Pipeline Workflow")
        # Render workflow graph in sidebar (vertical layout)
        render_workflow_graph(st.session_state.workflow_status, vertical=True)
        
        st.markdown("---")
        st.header("💬 Conversation History")
        if st.session_state.conversation_history:
            for idx, qa in enumerate(st.session_state.conversation_history[-5:], 1):  # Show last 5
                with st.expander(f"Q{idx}: {qa.get('question', '')[:50]}..."):
                    st.markdown(f"**Question:** {qa.get('question', '')}")
                    st.markdown(f"**Answer:** {qa.get('answer', '')[:200]}...")
        else:
            st.info("No conversation history yet.")
    
    # Load RAG system
    try:
        rag_system = load_rag_system()
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {str(e)}")
        st.stop()
    
    # Main interface
    st.markdown("### 🧪 Enter Your Question (Text) and/or Upload an Image")
    query = st.text_area(
        label="Question Input",
        label_visibility="hidden",  # Hide label but keep it for accessibility
        height=120,
        placeholder="Ask a science question...",
        help="The question will be automatically corrected, rephrased, and processed through the RAG pipeline. Ask follow-up questions to continue the conversation.",
        key="query_input"
    )
    uploaded_image = st.file_uploader(
        "Optional: Upload an image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Upload an image to trigger the image pipeline. If omitted, the text pipeline runs as before.",
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        process_button = st.button("🚀 Process Question", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear Conversation", use_container_width=True):
            clear_conversation()
            st.rerun()
    
    if process_button:
        # Determine input type based on presence of image
        has_image = uploaded_image is not None
        input_type = "image" if has_image else "text"

        if not has_image and not query.strip():
            st.warning("⚠️ Please enter a question or upload an image.")
            return

        # Save uploaded image temporarily (if any)
        # Resize image to reduce memory usage (max 1024x1024)
        image_path = None
        if has_image:
            import tempfile
            import io
            
            if PIL_AVAILABLE:
                # Read image
                img_bytes = uploaded_image.getbuffer()
                img = Image.open(io.BytesIO(img_bytes))
                
                # Resize if too large (max 512x512 to save memory - critical for Qwen3-vl:4b)
                max_size = 512  # Reduced from 1024 to save more memory
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    st.info(f"📐 Image resized to {img.width}x{img.height} to save memory")
                
                # Save to temp file as PNG
                suffix = ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    img.save(tmp.name, format='PNG')
                    image_path = tmp.name
            else:
                # Fallback: save without resizing (if PIL not available)
                suffix = Path(uploaded_image.name).suffix or ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_image.getbuffer())
                    image_path = tmp.name

        # Reset workflow status and set processing flag
        reset_workflow_status()
        st.session_state.processing = True
        st.session_state.current_query = query
        st.session_state.current_image_path = image_path
        st.session_state.current_input_type = input_type
        
        # Step 1: Detect and resolve follow-up questions (for text part)
        resolved_question = detect_followup_question(query, st.session_state.conversation_history) if query.strip() else query
        st.session_state.resolved_question = resolved_question if resolved_question else query
        
        # Update workflow status
        update_workflow_status("routing_node", "active")
        if has_image:
            update_workflow_status("image_processing_node", "active")
        
        # Rerun to process the query
        st.rerun()
    
    # Process the query if we have one and processing flag is set
    if (st.session_state.current_query or st.session_state.get("current_image_path")) and st.session_state.processing:
        query = st.session_state.current_query
        resolved = st.session_state.resolved_question
        image_path = st.session_state.get("current_image_path")
        input_type = st.session_state.get("current_input_type", "text")
        
        # Display original question
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### ✏️ Your Question")
        st.write(query if query else "Image-only query")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display resolved question if it's a follow-up
        if resolved and resolved != query:
            st.markdown('<div class="resolved-question-box">', unsafe_allow_html=True)
            st.markdown("### 🔗 Resolved Follow-up Question")
            st.info(resolved)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Execute pipeline steps sequentially
        update_workflow_status("routing_node", "active")
        
        with st.spinner("🔄 Processing question through RAG pipeline..."):
            try:
                # Execute full RAG pipeline (with image support)
                # Use resolved question if available, otherwise use original
                question_to_process = resolved if resolved else query
                result = rag_system.run(
                    question=question_to_process,
                    input_type=input_type,
                    image_path=image_path,
                )
                
                # Display performance report if available
                if "performance_report" in result:
                    perf_report = result["performance_report"]
                    with st.expander("📊 Performance Profiling Report", expanded=False):
                        st.markdown("### ⏱️ Total Execution Time")
                        total_time = perf_report.get("total_time_seconds", 0)
                        st.metric("Total Time", f"{total_time:.3f} seconds", f"{total_time*1000:.1f} ms")
                        
                        st.markdown("### 📋 Stage Breakdown")
                        stage_summary = perf_report.get("stage_summary", {})
                        if stage_summary:
                            # Create a table
                            import pandas as pd
                            stage_data = []
                            for stage_name, stats in sorted(stage_summary.items(), key=lambda x: x[1]['total_time'], reverse=True):
                                stage_data.append({
                                    "Stage": stage_name,
                                    "Total (s)": f"{stats['total_time']:.3f}",
                                    "Count": stats['count'],
                                    "Avg (s)": f"{stats['avg_time']:.3f}",
                                    "% of Total": f"{(stats['total_time'] / total_time * 100) if total_time > 0 else 0:.1f}%"
                                })
                            df = pd.DataFrame(stage_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Show slowest stage
                        slowest = perf_report.get("slowest_stage", {})
                        if slowest.get("name"):
                            st.markdown("### 🐌 Slowest Stage")
                            st.info(f"**{slowest['name']}**: {slowest['duration_seconds']:.3f}s ({(slowest['duration_seconds'] / total_time * 100) if total_time > 0 else 0:.1f}% of total)")
                
                # Update workflow status based on actual results
                update_workflow_status("routing_node", "completed")
                
                # Image processing status
                if image_path:
                    update_workflow_status("image_processing_node", "completed")
                    if result.get("input_type") == "image":
                        update_workflow_status("perception_routing_node", "completed")
                        if result.get("visual_description"):
                            update_workflow_status("visual_understanding_node", "completed")
                        if result.get("ocr_text"):
                            update_workflow_status("ocr_node", "completed")
                
                update_workflow_status("rephrase_question", "completed")
                
                # Split Question
                if result.get("question_parts") and len(result.get("question_parts", [])) > 0:
                    update_workflow_status("split_question", "completed")
                
                # Knowledge Retrieval
                if result.get("retrieved_docs") and len(result.get("retrieved_docs", [])) > 0:
                    update_workflow_status("kb_query_node", "completed")
                    # Presentation Control
                    if result.get("selected_images") is not None:
                        update_workflow_status("presentation_control_node", "completed")
                
                # Generate Answer
                if result.get("answer") and result.get("answer").strip():
                    update_workflow_status("generate_answer", "completed")
                
                # Store full pipeline state
                st.session_state.pipeline_state = {
                    "original_question": result.get("original_question", query),
                    "corrected_question": result.get("corrected_question", ""),
                    "rephrased_question": result.get("rephrased_question", ""),
                    "resolved_question": resolved if resolved != query else None,
                    "question_parts": result.get("question_parts", []),
                    "retrieved_docs_count": len(result.get("retrieved_docs", [])),
                    "retrieved_docs": [
                        {
                            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                            "score": float(score),
                            "metadata": doc.metadata
                        }
                        for doc, score in result.get("retrieved_docs", [])[:5]
                    ],
                    "answer": result.get("answer", ""),
                    "error": result.get("error", None),
                    "input_type": result.get("input_type", "text"),
                    "used_tools": result.get("used_tools", [])
                }
                
                # Display image path if available
                if image_path:
                    st.markdown('<div class="section-box">', unsafe_allow_html=True)
                    st.markdown("### 🖼️ Input Image")
                    st.info(f"**Image Path:** `{image_path}`")
                    try:
                        if PIL_AVAILABLE:
                            img = Image.open(image_path)
                            st.image(img, caption="Uploaded Image", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display image: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display selected images for presentation
                selected_images = result.get("selected_images", [])
                if selected_images and len(selected_images) > 0:
                    st.markdown('<div class="section-box">', unsafe_allow_html=True)
                    st.markdown("### 🖼️ Selected Images for Display")
                    cols = st.columns(min(3, len(selected_images)))
                    for idx, img_info in enumerate(selected_images[:3]):
                        with cols[idx]:
                            image_id = img_info.get("image_id", "")
                            doc_id = img_info.get("source_doc_id", "")
                            st.caption(f"Doc ID: {doc_id}")
                            try:
                                if os.path.exists(image_id):
                                    img = Image.open(image_id)
                                    st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                                    st.caption(f"Path: `{image_id}`")
                                else:
                                    st.warning(f"Image not found: {image_id}")
                            except Exception as e:
                                st.warning(f"Could not display: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif image_path and not selected_images:
                    st.markdown('<div class="section-box">', unsafe_allow_html=True)
                    st.markdown("### 🖼️ Selected Images")
                    st.info("No relevant images found in retrieved documents.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display visual description if available
                visual_description = result.get("visual_description", "")
                if visual_description:
                    st.markdown('<div class="section-box">', unsafe_allow_html=True)
                    st.markdown("### 👁️ Visual Understanding")
                    st.info(f"**Image Description:** {visual_description}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display OCR text if available
                ocr_text = result.get("ocr_text", "")
                if ocr_text:
                    st.markdown('<div class="section-box">', unsafe_allow_html=True)
                    st.markdown("### 📝 Extracted Text (OCR)")
                    st.info(f"**Extracted Text:** {ocr_text}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display retrieved context
                retrieved_docs = result.get("retrieved_docs", [])
                st.markdown('<div class="section-box">', unsafe_allow_html=True)
                st.markdown("### 📚 Retrieved Context")
                if retrieved_docs:
                    for i, (doc, score) in enumerate(retrieved_docs[:3], 1):  # Show top 3
                        with st.expander(f"📄 Chunk {i} (Similarity Score: {score:.4f})", expanded=(i == 1)):
                            st.markdown(doc.page_content)
                            st.caption(f"Index: {doc.metadata.get('index', 'N/A')}")
                            # Show image path if available in metadata
                            img_path = doc.metadata.get("image_path", "")
                            if img_path:
                                st.caption(f"📷 Image: `{img_path}`")
                else:
                    st.warning("No documents retrieved.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display final answer
                answer = result.get("answer", "No answer generated.")
                st.markdown('<div class="section-box">', unsafe_allow_html=True)
                st.markdown("### 💡 Final Answer")
                if answer:
                    st.success(answer)
                    
                    # Automatic evaluation using Qwen 2.5-VL-7B-Instruct (OpenRouter) with structured outputs
                    try:
                        with st.spinner("🔍 Evaluating answer using Qwen 2.5-VL-7B-Instruct (OpenRouter)..."):
                            # Format retrieved context for evaluation (separate text and image captions)
                            text_context, image_captions = format_retrieved_context_for_evaluation(
                                retrieved_docs=retrieved_docs,
                                max_docs=10,
                                max_chars_per_doc=500
                            )
                            
                            # Use Qwen 2.5-VL-7B-Instruct for evaluation
                            evaluation = evaluate_rag_with_mistral(
                                query=query if query else "Image-only query",
                                retrieved_context=text_context,
                                generated_answer=answer,
                                image_captions=image_captions
                            )
                            
                            # Extract results
                            context_relevance = evaluation.get("context_relevance", {})
                            answer_relevance = evaluation.get("answer_relevance", {})
                            groundedness = evaluation.get("groundedness", {})
                            
                            # Display evaluation results
                            st.markdown("---")
                            st.markdown("### 🏆 RAG Evaluation Results")
                            
                            # Three columns for the three criteria
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**📚 Context Relevance**")
                                score = context_relevance.get("score", "0")
                                explanation = context_relevance.get("explanation", "")
                                score_int = int(score) if score.isdigit() else 0
                                
                                if score_int == 3:
                                    st.success(f"✅ High Relevance ({score}/3)")
                                elif score_int == 2:
                                    st.info(f"✓ Medium Relevance ({score}/3)")
                                elif score_int == 1:
                                    st.warning(f"⚠ Low Relevance ({score}/3)")
                                else:
                                    st.error(f"❌ No Relevance ({score}/3)")
                                
                                with st.expander("📝 Reasoning"):
                                    st.write(explanation)
                            
                            with col2:
                                st.markdown("**💬 Answer Relevance**")
                                score = answer_relevance.get("score", "0")
                                explanation = answer_relevance.get("explanation", "")
                                score_int = int(score) if score.isdigit() else 0
                                
                                if score_int == 3:
                                    st.success(f"✅ High Relevance ({score}/3)")
                                elif score_int == 2:
                                    st.info(f"✓ Medium Relevance ({score}/3)")
                                elif score_int == 1:
                                    st.warning(f"⚠ Low Relevance ({score}/3)")
                                else:
                                    st.error(f"❌ No Relevance ({score}/3)")
                                
                                with st.expander("📝 Reasoning"):
                                    st.write(explanation)
                            
                            with col3:
                                st.markdown("**🔗 Groundedness**")
                                score = groundedness.get("score", "0")
                                explanation = groundedness.get("explanation", "")
                                score_int = int(score) if score.isdigit() else 0
                                
                                if score_int == 3:
                                    st.success(f"✅ High Groundedness ({score}/3)")
                                elif score_int == 2:
                                    st.info(f"✓ Medium Groundedness ({score}/3)")
                                elif score_int == 1:
                                    st.warning(f"⚠ Low Groundedness ({score}/3)")
                                else:
                                    st.error(f"❌ Not Grounded ({score}/3)")
                                
                                with st.expander("📝 Reasoning"):
                                    st.write(explanation)
                            
                            # Store evaluation in session state
                            st.session_state.last_evaluation = evaluation
                            
                            # Display summary
                            st.markdown("---")
                            avg_score = (
                                int(context_relevance.get("score", "0")) + 
                                int(answer_relevance.get("score", "0")) + 
                                int(groundedness.get("score", "0"))
                            ) / 3
                            st.metric("📊 Overall Average Score", f"{avg_score:.2f}/3.0")
                            
                    except ImportError as e:
                        st.warning(f"⚠️ Evaluation unavailable: {str(e)}")
                        st.info("💡 To enable evaluation, install openai: `pip install openai` and configure OpenRouter API key")
                    except Exception as e:
                        st.warning(f"⚠️ Evaluation failed: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                else:
                    st.warning("Answer not available.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "question": query,
                    "resolved_question": resolved if resolved != query else None,
                    "answer": answer,
                    "retrieved_docs_count": len(retrieved_docs),
                    "corrected_question": result.get("corrected_question", ""),
                    "rephrased_question": result.get("rephrased_question", "")
                })
                
                # Display workflow graph at bottom of page
                st.markdown("---")
                st.markdown("### 🔄 Current Pipeline Status")
                
                # Build metadata for workflow display
                workflow_metadata = {
                    "retrieved_docs_count": len(retrieved_docs)
                }
                
                render_workflow_graph(
                    st.session_state.workflow_status,
                    image_path=image_path,
                    metadata=workflow_metadata,
                    vertical=False
                )
                
                # State Graph Visualization
                st.markdown("---")
                st.markdown("### 📊 State Graph Visualization with Image Path Tracking")
                with st.expander("View Complete State Graph", expanded=False):
                    try:
                        from rag.state_graph_viz import render_state_graph
                        render_state_graph(result, image_path=image_path)
                    except ImportError:
                        st.warning("State graph visualization module not available.")
                    except Exception as e:
                        st.error(f"Error rendering state graph: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # Show detailed view in expander
                with st.expander("📊 Detailed Pipeline Information"):
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📝 Question Processing",
                        "🔍 Full Retrieval",
                        "🔧 Debug",
                        "📋 Full State (JSON)"
                    ])
                    
                    with tab1:
                        st.subheader("✏️ Original Question")
                        st.info(result.get("original_question", query))
                        
                        st.subheader("✅ Corrected Question")
                        corrected = result.get("corrected_question", "N/A")
                        if corrected and corrected != "N/A":
                            st.success(corrected)
                        else:
                            st.info("No correction needed or available.")
                        
                        st.subheader("🔄 Rephrased Question")
                        rephrased = result.get("rephrased_question", "N/A")
                        if rephrased and rephrased != "N/A":
                            st.success(rephrased)
                        else:
                            st.info("No rephrasing needed or available.")
                        
                        st.subheader("🧩 Decomposed Parts")
                        parts = result.get("question_parts", [])
                        if parts and len(parts) > 0:
                            for i, part in enumerate(parts, 1):
                                st.markdown(f"**Part {i}:** {part}")
                        else:
                            st.info("No decomposition needed or available.")
                    
                    with tab2:
                        st.subheader("📚 All Retrieved Documents")
                        if retrieved_docs:
                            for i, (doc, score) in enumerate(retrieved_docs, 1):
                                with st.expander(f"📄 Chunk {i} (Similarity Score: {score:.4f})", expanded=(i == 1)):
                                    st.markdown(doc.page_content)
                                    st.caption(f"Index: {doc.metadata.get('index', 'N/A')}")
                        else:
                            st.warning("No documents retrieved.")
                    
                    with tab3:
                        st.subheader("🔧 Debug Information")
                        st.json({
                            "original_question": result.get("original_question", ""),
                            "corrected_question": result.get("corrected_question", ""),
                            "rephrased_question": result.get("rephrased_question", ""),
                            "question_parts": result.get("question_parts", []),
                            "retrieved_docs_count": len(result.get("retrieved_docs", [])),
                            "answer_length": len(result.get("answer", "")),
                            "error": result.get("error", None),
                            "input_type": result.get("input_type", "text"),
                            "used_tools": result.get("used_tools", [])
                        })
                        
                        st.subheader("📈 Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Retrieved Docs", len(result.get("retrieved_docs", [])))
                        with col2:
                            st.metric("Question Parts", len(result.get("question_parts", [])))
                        with col3:
                            answer_len = len(result.get("answer", ""))
                            st.metric("Answer Length", f"{answer_len} chars")
                    
                    with tab4:
                        st.subheader("📋 Full Pipeline State (JSON)")
                        st.json(st.session_state.pipeline_state)
                
                # Clear processing flag and current query to prevent reprocessing
                st.session_state.processing = False
                st.session_state.current_query = ""
                st.session_state.resolved_question = ""
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                reset_workflow_status()
                st.session_state.processing = False
                st.session_state.current_query = ""
                st.session_state.pipeline_state = {"error": str(e)}
                return


if __name__ == "__main__":
    main()
