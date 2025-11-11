"""
Streamlit interface for RAGfolio
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import json
import sqlite3
import logging

try:
    from .config import DATABASE_PATH, DOCUMENTS_DIR
    from .retriever import DocumentRetriever
    from .llm_integration import LLMIntegration
    from .indexer import DocumentIndexer
except ImportError:
    # For direct execution
    from config import DATABASE_PATH, DOCUMENTS_DIR
    from retriever import DocumentRetriever
    from llm_integration import LLMIntegration
    from indexer import DocumentIndexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="RAGfolio - Personal Knowledge Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.search-box {
    margin-bottom: 1rem;
}

.result-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: #f9f9f9;
}

.result-header {
    font-weight: bold;
    font-size: 1.1rem;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}

.result-meta {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 0.5rem;
}

.score-badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: bold;
}

.high-score { background-color: #d4edda; color: #155724; }
.medium-score { background-color: #fff3cd; color: #856404; }
.low-score { background-color: #f8d7da; color: #721c24; }

.priority-high { color: #dc3545; font-weight: bold; }
.priority-medium { color: #fd7e14; }
.priority-low { color: #6c757d; }

.topic-tag {
    display: inline-block;
    background-color: #e7f3ff;
    color: #0056b3;
    padding: 0.1rem 0.4rem;
    border-radius: 12px;
    font-size: 0.8rem;
    margin-right: 0.3rem;
    margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


class RAGfolioApp:
    """Streamlit app for RAGfolio"""
    
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.llm = LLMIntegration()
        self.indexer = DocumentIndexer()
        
        # Initialize session state
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'last_query' not in st.session_state:
            st.session_state.last_query = ""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def run(self):
        """Run the Streamlit app"""
        st.markdown('<h1 class="main-header">🔍 RAGfolio</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Your Personal Knowledge Search & Chat Assistant</p>', unsafe_allow_html=True)
        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "💬 Chat", "📊 Analytics", "⚙️ Management"])
        
        with tab1:
            self.search_tab()
        
        with tab2:
            self.chat_tab()
        
        with tab3:
            self.analytics_tab()
        
        with tab4:
            self.management_tab()
    
    def create_sidebar(self):
        """Create the sidebar with filters and options"""
        st.sidebar.header("🎛️ Search Filters")
        
        # Search settings
        max_results = st.sidebar.slider("Max Results", 1, 50, 10)
        
        # Topic filter
        topics = self.get_available_topics()
        selected_topics = st.sidebar.multiselect("Topics", topics)
        
        # Priority filter
        selected_priorities = st.sidebar.multiselect(
            "Priority", 
            ["high", "medium", "low"],
            default=[]
        )
        
        # Date range filter
        st.sidebar.subheader("Date Range")
        use_date_filter = st.sidebar.checkbox("Filter by date range")
        
        date_after = None
        date_before = None
        
        if use_date_filter:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                date_after = st.date_input("After", value=datetime.now() - timedelta(days=365))
            with col2:
                date_before = st.date_input("Before", value=datetime.now())
        
        # Store filters in session state
        st.session_state.filters = {
            'max_results': max_results,
            'topics': selected_topics,
            'priority': selected_priorities,
            'date_after': date_after,
            'date_before': date_before
        }
        
        # Index stats
        st.sidebar.header("📊 Index Stats")
        stats = self.get_index_stats()
        if stats:
            st.sidebar.metric("Documents", stats.get('documents', 0))
            st.sidebar.metric("Chunks", stats.get('chunks', 0))
            st.sidebar.metric("Vectors", stats.get('faiss_vectors', 0))
    
    def search_tab(self):
        """Main search interface"""
        
        # Search box
        query = st.text_input(
            "Search your knowledge base:",
            placeholder="Enter your search query...",
            key="search_input",
            help="You can use filters like 'topic:meditation', 'priority:high', 'after:2024-01-01'"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            search_clicked = st.button("🔍 Search", type="primary")
        
        with col2:
            response_type = st.selectbox(
                "Response Type",
                ["search", "answer", "summary", "analysis"],
                help="Choose how to process the results"
            )
        
        # Perform search
        if search_clicked and query:
            with st.spinner("Searching..."):
                st.session_state.last_query = query
                st.session_state.search_results = self.perform_search(query)
        
        # Display results
        if st.session_state.search_results:
            st.header(f"Search Results for: '{st.session_state.last_query}'")
            
            # Generate LLM response if requested
            if response_type != "search" and st.session_state.search_results:
                with st.expander("🤖 AI Response", expanded=True):
                    with st.spinner("Generating AI response..."):
                        llm_response = self.llm.generate_response(
                            st.session_state.last_query,
                            st.session_state.search_results,
                            response_type
                        )
                    
                    if 'error' not in llm_response:
                        st.markdown(llm_response['response'])
                        
                        # Show sources
                        with st.expander("📚 Sources"):
                            for source in llm_response['sources'][:5]:
                                st.write(f"**{source['index']}.** {source['title']} ({source['file_name']})")
                    else:
                        st.error(f"Error generating response: {llm_response.get('error', 'Unknown error')}")
            
            # Display search results
            self.display_search_results(st.session_state.search_results)
        
        elif query and search_clicked:
            st.info("No results found. Try different keywords or check your filters.")
    
    def chat_tab(self):
        """Chat interface for conversational search"""
        st.header("💬 Chat with your Knowledge Base")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, (user_msg, bot_msg, sources) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div style="text-align: right; margin: 1rem 0;">
                    <div style="background-color: #007bff; color: white; padding: 0.5rem 1rem; border-radius: 15px; display: inline-block; max-width: 70%;">
                        {user_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Bot message
                st.markdown(f"""
                <div style="text-align: left; margin: 1rem 0;">
                    <div style="background-color: #f1f3f4; color: #333; padding: 0.5rem 1rem; border-radius: 15px; display: inline-block; max-width: 70%;">
                        {bot_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Sources
                if sources:
                    with st.expander(f"Sources for response {i+1}", expanded=False):
                        for source in sources[:3]:
                            st.write(f"• **{source['title']}** ({source['file_name']})")
        
        # Chat input
        chat_input = st.text_input("Ask a question about your documents:", key="chat_input")
        
        if st.button("Send", type="primary") and chat_input:
            with st.spinner("Thinking..."):
                # Search for relevant content
                search_results = self.perform_search(chat_input)
                
                # Generate response
                if search_results:
                    llm_response = self.llm.generate_response(
                        chat_input, 
                        search_results, 
                        "answer"
                    )
                    
                    if 'error' not in llm_response:
                        bot_response = llm_response['response']
                        sources = llm_response['sources'][:3]
                    else:
                        bot_response = f"Sorry, I encountered an error: {llm_response.get('error', 'Unknown error')}"
                        sources = []
                else:
                    bot_response = "I couldn't find relevant information to answer your question. Try rephrasing or check if the topic is covered in your documents."
                    sources = []
                
                # Add to chat history
                st.session_state.chat_history.append((chat_input, bot_response, sources))
                st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    def analytics_tab(self):
        """Analytics and insights about the knowledge base"""
        st.header("📊 Knowledge Base Analytics")
        
        # Get analytics data
        analytics = self.get_analytics_data()
        
        if not analytics:
            st.warning("No data available. Please index some documents first.")
            return
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", analytics['total_docs'])
        with col2:
            st.metric("Total Chunks", analytics['total_chunks'])
        with col3:
            st.metric("Unique Topics", len(analytics['topic_counts']))
        with col4:
            st.metric("Avg Chunks/Doc", f"{analytics['total_chunks'] / max(analytics['total_docs'], 1):.1f}")
        
        # Documents by priority
        col1, col2 = st.columns(2)
        
        with col1:
            if analytics['priority_counts']:
                fig_priority = px.pie(
                    values=list(analytics['priority_counts'].values()),
                    names=list(analytics['priority_counts'].keys()),
                    title="Documents by Priority"
                )
                st.plotly_chart(fig_priority)
        
        with col2:
            if analytics['topic_counts']:
                # Top 10 topics
                top_topics = dict(sorted(analytics['topic_counts'].items(), key=lambda x: x[1], reverse=True)[:10])
                fig_topics = px.bar(
                    x=list(top_topics.values()),
                    y=list(top_topics.keys()),
                    orientation='h',
                    title="Top Topics"
                )
                fig_topics.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_topics)
        
        # Document timeline
        if analytics['doc_timeline']:
            df_timeline = pd.DataFrame([
                {'date': date, 'count': count} 
                for date, count in analytics['doc_timeline'].items()
            ])
            df_timeline['date'] = pd.to_datetime(df_timeline['date'])
            
            fig_timeline = px.line(
                df_timeline, 
                x='date', 
                y='count', 
                title="Documents Added Over Time"
            )
            st.plotly_chart(fig_timeline)
        
        # Recent activity
        st.subheader("Recent Activity")
        recent_docs = analytics.get('recent_docs', [])
        if recent_docs:
            df_recent = pd.DataFrame(recent_docs)
            st.dataframe(df_recent, use_container_width=True)
        else:
            st.info("No recent document activity.")
    
    def management_tab(self):
        """Document management and indexing"""
        st.header("⚙️ Document Management")
        
        # Indexing section
        st.subheader("📂 Indexing")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reindex All", help="Reprocess all documents"):
                with st.spinner("Reindexing all documents..."):
                    try:
                        self.indexer.index_all_documents(force_reprocess=True)
                        st.success("✅ Reindexing complete!")
                    except Exception as e:
                        st.error(f"❌ Error during reindexing: {e}")
        
        with col2:
            if st.button("🆕 Index New", help="Index only new or modified documents"):
                with st.spinner("Indexing new documents..."):
                    try:
                        self.indexer.index_all_documents(force_reprocess=False)
                        st.success("✅ New documents indexed!")
                    except Exception as e:
                        st.error(f"❌ Error during indexing: {e}")
        
        with col3:
            if st.button("🗑️ Rebuild Index", help="Clear and rebuild entire index"):
                if st.checkbox("I understand this will delete all current index data"):
                    with st.spinner("Rebuilding index..."):
                        try:
                            self.indexer.rebuild_index()
                            st.success("✅ Index rebuilt successfully!")
                        except Exception as e:
                            st.error(f"❌ Error rebuilding index: {e}")
        
        # Document status
        st.subheader("📄 Document Status")
        
        doc_status = self.get_document_status()
        if doc_status:
            df_status = pd.DataFrame([
                {
                    'File': Path(path).name,
                    'Status': 'Up to date' if not info['needs_processing'] else 'Needs processing',
                    'Last Processed': info['last_processed'] or 'Never',
                    'Version': info['version_info']['version_id'][:8],
                    'Processing Count': info['processing_count']
                }
                for path, info in doc_status.items()
            ])
            
            st.dataframe(df_status, use_container_width=True)
        else:
            st.info("No documents found in the documents directory.")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        stats = self.get_index_stats()
        if stats:
            st.json(stats)
        
        # File upload
        st.subheader("📤 Add Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=['txt', 'md', 'pdf', 'docx'],
            help="Upload documents to add to your knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Save uploaded file
                    file_path = DOCUMENTS_DIR / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Index the file
                    try:
                        self.indexer.add_document(file_path, force_reprocess=True)
                        st.success(f"✅ Processed: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.success("🎉 All files processed!")
    
    def perform_search(self, query: str):
        """Perform search with current filters"""
        filters = st.session_state.filters
        
        # Build filter dict for retriever
        search_filters = {}
        
        if filters['topics']:
            search_filters['topics'] = filters['topics']
        
        if filters['priority']:
            search_filters['priority'] = filters['priority']
        
        if filters['date_after'] or filters['date_before']:
            date_range = {}
            if filters['date_after']:
                date_range['after'] = filters['date_after'].isoformat()
            if filters['date_before']:
                date_range['before'] = filters['date_before'].isoformat()
            search_filters['date_range'] = date_range
        
        return self.retriever.search(
            query, 
            max_results=filters['max_results'], 
            filters=search_filters
        )
    
    def display_search_results(self, results):
        """Display search results in a nice format"""
        for result in results:
            with st.container():
                # Result header
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{result['rank']}. {result['title']}**")
                
                with col2:
                    score_class = "high-score" if result['final_score'] > 0.7 else "medium-score" if result['final_score'] > 0.4 else "low-score"
                    st.markdown(f'<span class="score-badge {score_class}">Score: {result["final_score"]:.3f}</span>', unsafe_allow_html=True)
                
                with col3:
                    priority_class = f"priority-{result['priority']}"
                    st.markdown(f'<span class="{priority_class}">●</span> {result["priority"].title()}', unsafe_allow_html=True)
                
                # Metadata
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.caption(f"📁 {result['file_name']} • 📅 {result['modified_time'][:10]} • 🔧 {result['days_old']} days old")
                
                with col2:
                    # Topics
                    topics_html = "".join([f'<span class="topic-tag">{topic}</span>' for topic in result['topics'][:3]])
                    if topics_html:
                        st.markdown(topics_html, unsafe_allow_html=True)
                
                # Content preview
                content = result['content']
                if len(content) > 300:
                    content = content[:300] + "..."
                
                st.markdown(f"*{content}*")
                
                # Detailed scores in expander
                with st.expander("Score Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Similarity", f"{result['similarity_score']:.3f}")
                    with col2:
                        st.metric("Recency", f"{result['recency_score']:.3f}")
                    with col3:
                        st.metric("Priority", f"{result['priority_score']:.3f}")
                
                st.divider()
    
    def get_available_topics(self):
        """Get all available topics from the database"""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT topics FROM documents WHERE topics != ''")
                
                all_topics = set()
                for row in cursor.fetchall():
                    if row[0]:
                        topics = json.loads(row[0])
                        all_topics.update(topics)
                
                return sorted(list(all_topics))
        except Exception:
            return []
    
    def get_index_stats(self):
        """Get index statistics"""
        try:
            return self.indexer.get_index_stats()
        except Exception:
            return None
    
    def get_analytics_data(self):
        """Get analytics data for the analytics tab"""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM documents")
                total_docs = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0]
                
                if total_docs == 0:
                    return None
                
                # Priority distribution
                cursor.execute("SELECT priority, COUNT(*) FROM documents GROUP BY priority")
                priority_counts = dict(cursor.fetchall())
                
                # Topic counts
                cursor.execute("SELECT topics FROM documents WHERE topics != ''")
                topic_counts = {}
                for row in cursor.fetchall():
                    if row[0]:
                        topics = json.loads(row[0])
                        for topic in topics:
                            topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                # Document timeline (by month)
                cursor.execute("""
                    SELECT DATE(modified_time, 'start of month') as month, COUNT(*)
                    FROM documents 
                    GROUP BY month 
                    ORDER BY month DESC 
                    LIMIT 12
                """)
                doc_timeline = dict(cursor.fetchall())
                
                # Recent documents
                cursor.execute("""
                    SELECT file_name, title, modified_time, priority
                    FROM documents 
                    ORDER BY processed_time DESC 
                    LIMIT 10
                """)
                recent_docs = [
                    {
                        'File': row[0],
                        'Title': row[1],
                        'Modified': row[2][:10],
                        'Priority': row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
                return {
                    'total_docs': total_docs,
                    'total_chunks': total_chunks,
                    'priority_counts': priority_counts,
                    'topic_counts': topic_counts,
                    'doc_timeline': doc_timeline,
                    'recent_docs': recent_docs
                }
                
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return None
    
    def get_document_status(self):
        """Get document processing status"""
        try:
            from .config import EMBEDDING_MODEL
            return self.indexer.version_manager.get_all_documents_status(DOCUMENTS_DIR, EMBEDDING_MODEL)
        except Exception:
            return None


def main():
    """Main function to run the Streamlit app"""
    app = RAGfolioApp()
    app.run()


if __name__ == "__main__":
    main()
