"""
Version management for documents using Git and file timestamps
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    from .config import VERSIONS_DIR, USE_GIT_VERSIONING
except ImportError:
    # For direct execution
    from config import VERSIONS_DIR, USE_GIT_VERSIONING

logger = logging.getLogger(__name__)


class VersionManager:
    """Manage document versions using Git or file-based tracking"""
    
    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self.versions_file = VERSIONS_DIR / "versions.json"
        self.use_git = USE_GIT_VERSIONING and GIT_AVAILABLE
        self.repo = None
        
        if self.use_git:
            try:
                self.repo = git.Repo(self.repo_path, search_parent_directories=True)
            except (git.InvalidGitRepositoryError, git.GitCommandError):
                logger.warning("Git repository not found, using file-based versioning")
                self.use_git = False
        
        # Ensure versions directory exists
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize versions file if it doesn't exist
        if not self.versions_file.exists():
            self._save_versions({})
    
    def get_document_version(self, file_path: Path) -> Dict[str, Any]:
        """
        Get version information for a document
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary with version information
        """
        if self.use_git:
            return self._get_git_version(file_path)
        else:
            return self._get_file_version(file_path)
    
    def _get_git_version(self, file_path: Path) -> Dict[str, Any]:
        """Get version info using Git"""
        try:
            # Get the latest commit that modified this file
            commits = list(self.repo.iter_commits(paths=str(file_path), max_count=1))
            
            if commits:
                commit = commits[0]
                version_info = {
                    'version_id': commit.hexsha[:8],
                    'full_hash': commit.hexsha,
                    'author': commit.author.name,
                    'date': commit.committed_datetime.isoformat(),
                    'message': commit.message.strip(),
                    'type': 'git'
                }
            else:
                # File not in git history, use file stats
                version_info = self._get_file_version(file_path)
                version_info['type'] = 'file_fallback'
            
            return version_info
            
        except Exception as e:
            logger.warning(f"Error getting Git version for {file_path}: {e}")
            return self._get_file_version(file_path)
    
    def _get_file_version(self, file_path: Path) -> Dict[str, Any]:
        """Get version info using file system"""
        stat = file_path.stat()
        
        # Create a hash based on file content for version ID
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        return {
            'version_id': file_hash,
            'full_hash': hashlib.md5(str(file_path).encode() + str(stat.st_mtime).encode()).hexdigest(),
            'date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'size': stat.st_size,
            'type': 'file'
        }
    
    def get_document_history(self, file_path: Path, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get version history for a document
        
        Args:
            file_path: Path to the document
            limit: Maximum number of versions to return
            
        Returns:
            List of version dictionaries, newest first
        """
        if self.use_git:
            return self._get_git_history(file_path, limit)
        else:
            return self._get_file_history(file_path, limit)
    
    def _get_git_history(self, file_path: Path, limit: int) -> List[Dict[str, Any]]:
        """Get Git history for a file"""
        try:
            commits = list(self.repo.iter_commits(paths=str(file_path), max_count=limit))
            
            history = []
            for commit in commits:
                history.append({
                    'version_id': commit.hexsha[:8],
                    'full_hash': commit.hexsha,
                    'author': commit.author.name,
                    'date': commit.committed_datetime.isoformat(),
                    'message': commit.message.strip(),
                    'type': 'git'
                })
            
            return history
            
        except Exception as e:
            logger.warning(f"Error getting Git history for {file_path}: {e}")
            return [self._get_file_version(file_path)]
    
    def _get_file_history(self, file_path: Path, limit: int) -> List[Dict[str, Any]]:
        """Get file-based history (limited to current version)"""
        return [self._get_file_version(file_path)]
    
    def track_document_processing(self, file_path: Path, chunk_count: int, 
                                embedding_model: str) -> str:
        """
        Track when a document was processed for indexing
        
        Args:
            file_path: Path to the processed document
            chunk_count: Number of chunks created
            embedding_model: Model used for embeddings
            
        Returns:
            Processing ID
        """
        version_info = self.get_document_version(file_path)
        processing_id = f"{version_info['version_id']}_{int(datetime.now().timestamp())}"
        
        processing_record = {
            'processing_id': processing_id,
            'file_path': str(file_path),
            'version_info': version_info,
            'processed_at': datetime.now().isoformat(),
            'chunk_count': chunk_count,
            'embedding_model': embedding_model
        }
        
        # Load existing versions
        versions = self._load_versions()
        
        # Add processing record
        file_key = str(file_path)
        if file_key not in versions:
            versions[file_key] = {'processing_history': []}
        
        versions[file_key]['processing_history'].append(processing_record)
        
        # Keep only last 10 processing records per file
        versions[file_key]['processing_history'] = versions[file_key]['processing_history'][-10:]
        
        self._save_versions(versions)
        return processing_id
    
    def get_processing_history(self, file_path: Path) -> List[Dict[str, Any]]:
        """Get processing history for a document"""
        versions = self._load_versions()
        file_key = str(file_path)
        
        if file_key in versions and 'processing_history' in versions[file_key]:
            return versions[file_key]['processing_history']
        
        return []
    
    def needs_reprocessing(self, file_path: Path, embedding_model: str) -> bool:
        """
        Check if a document needs reprocessing based on version changes
        
        Args:
            file_path: Path to the document
            embedding_model: Current embedding model
            
        Returns:
            True if document needs reprocessing
        """
        current_version = self.get_document_version(file_path)
        processing_history = self.get_processing_history(file_path)
        
        if not processing_history:
            return True  # Never processed
        
        latest_processing = processing_history[-1]
        
        # Check if version changed
        if latest_processing['version_info']['version_id'] != current_version['version_id']:
            return True
        
        # Check if embedding model changed
        if latest_processing['embedding_model'] != embedding_model:
            return True
        
        return False
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load versions from file"""
        try:
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_versions(self, versions: Dict[str, Any]) -> None:
        """Save versions to file"""
        with open(self.versions_file, 'w') as f:
            json.dump(versions, f, indent=2, default=str)
    
    def get_all_documents_status(self, documents_dir: Path, 
                               embedding_model: str) -> Dict[str, Dict[str, Any]]:
        """
        Get processing status for all documents
        
        Args:
            documents_dir: Directory containing documents
            embedding_model: Current embedding model
            
        Returns:
            Dictionary mapping file paths to status info
        """
        status = {}
        
        # Find all supported documents
        for file_path in documents_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in {'.md', '.txt', '.docx', '.pdf'}:
                version_info = self.get_document_version(file_path)
                needs_processing = self.needs_reprocessing(file_path, embedding_model)
                processing_history = self.get_processing_history(file_path)
                
                status[str(file_path)] = {
                    'version_info': version_info,
                    'needs_processing': needs_processing,
                    'last_processed': processing_history[-1]['processed_at'] if processing_history else None,
                    'processing_count': len(processing_history)
                }
        
        return status
