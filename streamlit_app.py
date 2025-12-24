"""
Solar Panel Fault Detection System
===================================
Simple fault detection and management
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random

# Page config
st.set_page_config(
    page_title="Solar Fault Detection",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AGGRESSIVE CSS FIXES - Force all text to be visible
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #0a0e1a !important;
        color: #ffffff !important;
    }
    
    /* Force ALL text to white */
    * {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, a {
        color: #ffffff !important;
    }
    
    /* Top menu bar fix */
    header[data-testid="stHeader"] {
        background-color: #1a1f2e !important;
    }
    
    /* Toolbar buttons */
    button[kind="header"] {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e !important;
    }
    
    /* ==== CRITICAL SELECTBOX FIXES ==== */
    
    /* Selectbox label */
    .stSelectbox label {
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Selectbox container */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1f2e !important;
    }
    
    /* Selectbox control (the actual dropdown button) */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1a1f2e !important;
        border: 1px solid #4b5563 !important;
    }
    
    /* Selected value text */
    .stSelectbox div[data-baseweb="select"] > div > div {
        color: #ffffff !important;
    }
    
    /* Dropdown arrow */
    .stSelectbox svg {
        fill: #ffffff !important;
    }
    
    /* Dropdown menu popup */
    div[data-baseweb="popover"] {
        background-color: #1a1f2e !important;
    }
    
    div[data-baseweb="popover"] > div {
        background-color: #1a1f2e !important;
    }
    
    /* Dropdown options container */
    ul[role="listbox"] {
        background-color: #1a1f2e !important;
    }
    
    /* Individual dropdown options */
    li[role="option"] {
        background-color: #1a1f2e !important;
        color: #ffffff !important;
    }
    
    li[role="option"]:hover {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    /* Option text */
    li[role="option"] * {
        color: #ffffff !important;
    }
    
    /* ==== RADIO BUTTON FIXES ==== */
    
    .stRadio label {
        color: #ffffff !important;
    }
    
    .stRadio div[role="radiogroup"] {
        background-color: #1a1f2e !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
    }
    
    .stRadio div[role="radio"] label {
        color: #ffffff !important;
    }
    
    /* ==== BUTTON FIXES ==== */
    
    .stButton > button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background-color: #2563eb !important;
    }
    
    .stButton > button * {
        color: #ffffff !important;
    }
    
    /* ==== FILE UPLOADER FIXES ==== */
    
    .stFileUploader {
        border: 2px dashed #4b5563 !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        background-color: #1a1f2e !important;
    }
    
    .stFileUploader label {
        color: #ffffff !important;
    }
    
    .stFileUploader section {
        background-color: #1a1f2e !important;
    }
    
    /* ==== EXPANDER FIXES ==== */
    
    .streamlit-expanderHeader {
        background-color: #1a1f2e !important;
        color: #ffffff !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1f2e !important;
    }
    
    details[data-testid="stExpander"] {
        background-color: #1a1f2e !important;
    }
    
    /* ==== ALERT BOXES ==== */
    
    .stAlert {
        color: #ffffff !important;
        background-color: #1a1f2e !important;
    }
    
    .stAlert * {
        color: #ffffff !important;
    }
    
    /* ==== METRICS ==== */
    
    .stMetric {
        background-color: #1a1f2e !important;
    }
    
    .stMetric label {
        color: #e5e7eb !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* ==== DATAFRAME ==== */
    
    .stDataFrame {
        background-color: #1a1f2e !important;
    }
    
    /* ==== PROGRESS BAR ==== */
    
    .stProgress > div > div {
        background-color: #3b82f6 !important;
    }
    
    /* ==== MARKDOWN ==== */
    
    .stMarkdown {
        color: #ffffff !important;
    }
    
    .stMarkdown * {
        color: #ffffff !important;
    }
    
    /* ==== INPUT FIELDS ==== */
    
    input {
        background-color: #1a1f2e !important;
        color: #ffffff !important;
        border: 1px solid #4b5563 !important;
    }
    
    textarea {
        background-color: #1a1f2e !important;
        color: #ffffff !important;
        border: 1px solid #4b5563 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Fault info
FAULT_INFO = {
    'Cell': {'severity': 'High', 'icon': '⚡', 'loss': '5-15%', 'action': 'Inspect cell, check connections'},
    'Cell-Multi': {'severity': 'Critical', 'icon': '🔥', 'loss': '20-40%', 'action': 'Replace module immediately'},
    'Cracking': {'severity': 'Medium', 'icon': '💔', 'loss': '3-10%', 'action': 'Monitor and schedule replacement'},
    'Diode': {'severity': 'High', 'icon': '⚙️', 'loss': '10-25%', 'action': 'Replace bypass diode'},
    'Diode-Multi': {'severity': 'Critical', 'icon': '🚨', 'loss': '30-50%', 'action': 'Emergency diode replacement'},
    'Hot-Spot': {'severity': 'High', 'icon': '🔥', 'loss': '15-30%', 'action': 'Check for shading, replace if needed'},
    'Hot-Spot-Multi': {'severity': 'Critical', 'icon': '🚨', 'loss': '40-70%', 'action': 'URGENT: Disconnect and replace'},
    'No-Anomaly': {'severity': 'Low', 'icon': '✅', 'loss': '0%', 'action': 'Continue routine monitoring'},
    'Offline-Module': {'severity': 'Critical', 'icon': '⚠️', 'loss': '100%', 'action': 'Check connections, test output'},
    'Shadowing': {'severity': 'Medium', 'icon': '🌳', 'loss': '10-30%', 'action': 'Remove shading source'},
    'Soiling': {'severity': 'Low', 'icon': '🧹', 'loss': '2-8%', 'action': 'Clean panels'},
    'Vegetation': {'severity': 'Medium', 'icon': '🌱', 'loss': '5-20%', 'action': 'Remove vegetation'}
}

@st.cache_resource
def load_model():
    try:
        return YOLO('runs/classify/solar_fault_detection/weights/best.pt')
    except:
        return None

# Initialize fault database
if 'fault_database' not in st.session_state:
    st.session_state.fault_database = pd.DataFrame({
        'Panel ID': ['A-125', 'B-087', 'C-234', 'A-089', 'D-156'],
        'Fault Type': ['Hot-Spot', 'Cell', 'Diode', 'Cracking', 'Soiling'],
        'Severity': ['High', 'High', 'Medium', 'Medium', 'Low'],
        'Detected': [
            (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(hours=10)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(hours=12)).strftime('%Y-%m-%d %H:%M')
        ],
        'Assigned To': ['John Smith', 'Sarah Johnson', 'Mike Chen', 'John Smith', 'Sarah Johnson'],
        'Status': ['In Progress', 'Pending', 'Assigned', 'Completed', 'Pending'],
        'Efficiency Loss': ['15-30%', '5-15%', '10-25%', '3-10%', '2-8%']
    })

# Sidebar Navigation
with st.sidebar:
    st.markdown("# 🌞 Solar Fault System")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🔍 Analyze Image", "📋 Fault Management", "📊 System Overview"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    st.info("""
    **Model:** YOLOv8n-cls  
    **Accuracy:** 75.88%  
    **Classes:** 12 fault types
    """)
    
    st.markdown("---")
    st.markdown("### 📞 Quick Stats")
    total = len(st.session_state.fault_database)
    active = len(st.session_state.fault_database[st.session_state.fault_database['Status'] != 'Completed'])
    st.metric("Total Faults", total)
    st.metric("Active", active)

# ==========================================
# PAGE 1: ANALYZE IMAGE
# ==========================================
if page == "🔍 Analyze Image":
    st.markdown("# 🔍 Thermal Image Analysis")
    st.markdown("### Upload thermal images for AI-powered fault detection")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload section
        st.markdown("### 📤 Upload Thermal Image")
        uploaded_file = st.file_uploader(
            "Choose a thermal image of a solar panel",
            type=['jpg', 'jpeg', 'png']
        )
    
    with col2:
        # Sample images dropdown
        st.markdown("### 🖼️ Or Try Sample")
        
        sample_dir = Path('sample_images')
        selected_sample = None
        
        if sample_dir.exists():
            samples = list(sample_dir.glob('*.jpg'))
            
            if samples:
                sample_names = ["-- Select Sample Image --"] + [img.name for img in samples]
                
                selected = st.selectbox(
                    "Sample Images",
                    sample_names,
                    key="sample_selector"
                )
                
                if selected != "-- Select Sample Image --":
                    selected_sample = sample_dir / selected
                    
                    # Show preview
                    st.image(Image.open(selected_sample), caption="Preview", use_container_width=True)
    
    # Process image
    image_to_process = None
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        st.success("✅ Image uploaded successfully!")
    elif selected_sample is not None:
        image_to_process = Image.open(selected_sample)
        st.info(f"📸 Using sample: {selected_sample.name}")
    
    if image_to_process is not None:
        model = load_model()
        
        if model:
            with st.spinner('🔄 Analyzing thermal image...'):
                results = model.predict(image_to_process, verbose=False)
            
            top_idx = results[0].probs.top1
            top_class = results[0].names[top_idx]
            top_conf = results[0].probs.top1conf.item()
            
            info = FAULT_INFO[top_class]
            
            st.markdown("---")
            st.markdown("## 📊 DETECTION REPORT")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📸 Analyzed Image")
                st.image(image_to_process, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Detection Result")
                
                # Result box
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1a1f2e, #252d3d); 
                            padding: 2rem; border-radius: 10px; text-align: center;
                            border: 3px solid #3b82f6; margin: 1rem 0;'>
                    <h1 style='color: white; margin: 0;'>{info['icon']} {top_class}</h1>
                    <h2 style='color: #9ca3af; margin: 1rem 0;'>Confidence: {top_conf*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Severity
                severity_colors = {'Critical': '#ef4444', 'High': '#f59e0b', 'Medium': '#eab308', 'Low': '#10b981'}
                color = severity_colors[info['severity']]
                
                st.markdown(f"""
                <div style='background: {color}; padding: 1.5rem; border-radius: 8px; 
                            text-align: center; margin: 1rem 0;'>
                    <h2 style='margin: 0; color: white;'>⚠️ {info['severity']} Severity</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**💰 Efficiency Loss:** {info['loss']}")
                st.markdown(f"**🔧 Action Required:** {info['action']}")
            
            # Details section
            st.markdown("---")
            st.markdown("### 📄 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Detection Information:**")
                st.write(f"• **Fault Type:** {top_class}")
                st.write(f"• **Severity:** {info['severity']}")
                st.write(f"• **Confidence:** {top_conf*100:.2f}%")
                st.write(f"• **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.markdown("**Impact Assessment:**")
                st.write(f"• **Efficiency Loss:** {info['loss']}")
                st.write(f"• **Recommended Action:** {info['action']}")
                st.write(f"• **Icon:** {info['icon']}")
            
            # Actions (Confidence Distribution section REMOVED)
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                report = f"""SOLAR PANEL FAULT DETECTION REPORT
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DETECTION RESULT
-----------------
Fault Type: {top_class}
Confidence: {top_conf*100:.2f}%
Severity: {info['severity']}

IMPACT
------
Efficiency Loss: {info['loss']}
Action: {info['action']}
"""
                
                st.download_button(
                    "📄 Download Report",
                    report,
                    f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True
                )
            
            with col2:
                if st.button("➕ Add to Database", use_container_width=True):
                    new_fault = pd.DataFrame({
                        'Panel ID': [f"{random.choice(['A', 'B', 'C', 'D'])}-{random.randint(100, 999)}"],
                        'Fault Type': [top_class],
                        'Severity': [info['severity']],
                        'Detected': [datetime.now().strftime('%Y-%m-%d %H:%M')],
                        'Assigned To': ['Unassigned'],
                        'Status': ['New'],
                        'Efficiency Loss': [info['loss']]
                    })
                    st.session_state.fault_database = pd.concat([new_fault, st.session_state.fault_database], ignore_index=True)
                    st.success("✅ Added to database!")
            
            with col3:
                if st.button("🔄 Analyze Another", use_container_width=True):
                    st.rerun()

# ==========================================
# PAGE 2: FAULT MANAGEMENT
# ==========================================
elif page == "📋 Fault Management":
    st.markdown("# 📋 Fault Management System")
    st.markdown("### Track and manage detected faults across your solar farm")
    st.markdown("---")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filter_status = st.selectbox("🔍 Filter by Status", ["All", "New", "Assigned", "In Progress", "Pending", "Completed"])
    
    with col2:
        filter_severity = st.selectbox("⚠️ Filter by Severity", ["All", "Critical", "High", "Medium", "Low"])
    
    with col3:
        filter_assigned = st.selectbox("👤 Filter by Technician", 
                                       ["All", "Unassigned", "John Smith", "Sarah Johnson", "Mike Chen", "Emma Davis"])
    
    with col4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Filter data
    filtered_df = st.session_state.fault_database.copy()
    
    if filter_status != "All":
        filtered_df = filtered_df[filtered_df['Status'] == filter_status]
    
    if filter_severity != "All":
        filtered_df = filtered_df[filtered_df['Severity'] == filter_severity]
    
    if filter_assigned != "All":
        filtered_df = filtered_df[filtered_df['Assigned To'] == filter_assigned]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Faults", len(filtered_df))
    
    with col2:
        critical = len(filtered_df[filtered_df['Severity'] == 'Critical'])
        st.metric("🔴 Critical", critical)
    
    with col3:
        in_progress = len(filtered_df[filtered_df['Status'] == 'In Progress'])
        st.metric("⚙️ In Progress", in_progress)
    
    with col4:
        completed = len(filtered_df[filtered_df['Status'] == 'Completed'])
        st.metric("✅ Completed", completed)
    
    st.markdown("---")
    
    # Fault cards display
    st.markdown("### 🗂️ Fault Records")
    
    if len(filtered_df) == 0:
        st.info("No faults match the selected filters")
    else:
        for idx, row in filtered_df.iterrows():
            with st.expander(f"📍 **{row['Panel ID']}** - {FAULT_INFO[row['Fault Type']]['icon']} {row['Fault Type']} ({row['Severity']})", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Fault Details:**")
                    st.write(f"• **Panel ID:** {row['Panel ID']}")
                    st.write(f"• **Fault Type:** {row['Fault Type']}")
                    st.write(f"• **Severity:** {row['Severity']}")
                    st.write(f"• **Efficiency Loss:** {row['Efficiency Loss']}")
                    st.write(f"• **Detected:** {row['Detected']}")
                
                with col2:
                    st.markdown("**Assignment & Status:**")
                    
                    # Editable fields
                    new_assigned = st.selectbox(
                        "👤 Assign Technician:",
                        ["Unassigned", "John Smith", "Sarah Johnson", "Mike Chen", "Emma Davis"],
                        index=["Unassigned", "John Smith", "Sarah Johnson", "Mike Chen", "Emma Davis"].index(row['Assigned To']),
                        key=f"assign_{idx}"
                    )
                    
                    new_status = st.selectbox(
                        "📊 Update Status:",
                        ["New", "Assigned", "In Progress", "Pending", "Completed"],
                        index=["New", "Assigned", "In Progress", "Pending", "Completed"].index(row['Status']),
                        key=f"status_{idx}"
                    )
                    
                    if st.button("💾 Save Changes", key=f"save_{idx}"):
                        st.session_state.fault_database.at[idx, 'Assigned To'] = new_assigned
                        st.session_state.fault_database.at[idx, 'Status'] = new_status
                        st.success("✅ Updated!")
                        st.rerun()

# ==========================================
# PAGE 3: SYSTEM OVERVIEW
# ==========================================
elif page == "📊 System Overview":
    st.markdown("# 📊 System Overview & Analytics")
    st.markdown("### Monitor system-wide performance and metrics")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    df = st.session_state.fault_database
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1f2e, #252d3d); padding: 1.5rem; 
                    border-radius: 10px; text-align: center; border-left: 4px solid #3b82f6;'>
            <h3 style='color: #9ca3af; margin: 0;'>Total Panels</h3>
            <h1 style='color: white; margin: 0.5rem 0;'>1,247</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        active = len(df[df['Status'] != 'Completed'])
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1f2e, #252d3d); padding: 1.5rem; 
                    border-radius: 10px; text-align: center; border-left: 4px solid #ef4444;'>
            <h3 style='color: #9ca3af; margin: 0;'>Active Faults</h3>
            <h1 style='color: white; margin: 0.5rem 0;'>{active}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1f2e, #252d3d); padding: 1.5rem; 
                    border-radius: 10px; text-align: center; border-left: 4px solid #10b981;'>
            <h3 style='color: #9ca3af; margin: 0;'>Avg Response</h3>
            <h1 style='color: white; margin: 0.5rem 0;'>2.3 hrs</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        health = 100 - (active / 12.47)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1f2e, #252d3d); padding: 1.5rem; 
                    border-radius: 10px; text-align: center; border-left: 4px solid #8b5cf6;'>
            <h3 style='color: #9ca3af; margin: 0;'>System Health</h3>
            <h1 style='color: white; margin: 0.5rem 0;'>{health:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Fault Type Distribution")
        fault_counts = df['Fault Type'].value_counts()
        st.bar_chart(fault_counts)
    
    with col2:
        st.markdown("### ⚠️ Severity Levels")
        severity_counts = df['Severity'].value_counts()
        st.bar_chart(severity_counts)
    
    st.markdown("---")
    
    # Team workload
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 Technician Workload")
        workload = df['Assigned To'].value_counts().reset_index()
        workload.columns = ['Technician', 'Assigned Faults']
        st.dataframe(workload, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📈 Status Breakdown")
        status = df['Status'].value_counts().reset_index()
        status.columns = ['Status', 'Count']
        st.dataframe(status, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # System info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 AI Model Information")
        st.info("""
        **Model:** YOLOv8n-cls  
        **Accuracy:** 75.88%  
        **Top-5 Accuracy:** 98.46%  
        **Classes:** 12 fault types  
        **Training Data:** 20,000+ images
        """)
    
    with col2:
        st.markdown("### 💻 System Status")
        st.success("""
        **Status:** ✅ Online  
        **Last Update:** Just now  
        **Uptime:** 99.7%  
        **Version:** 1.0.0  
        **Last Scan:** 5 minutes ago
        """)