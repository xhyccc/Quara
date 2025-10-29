"""
Zettelkasten-based Long-Term Memory System for QuARA
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import uuid
import json
from abc import ABC, abstractmethod

# Vector database and similarity search
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class MemoryNode:
    """
    Atomic memory node representing a discrete piece of knowledge
    Based on Zettelkasten principles
    """
    
    def __init__(self, 
                 content: str,
                 node_type: str,
                 metadata: Dict[str, Any] = None,
                 node_id: str = None):
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:12]}"
        self.content = content
        self.node_type = node_type  # hypothesis, dataset, result, method, etc.
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.links = []  # Links to other nodes
        self.tags = set()
        self.importance_score = 0.0
    
    def add_link(self, target_node_id: str, link_type: str = "related"):
        """Add semantic link to another node"""
        link = {
            "target": target_node_id,
            "type": link_type,
            "created_at": datetime.now()
        }
        self.links.append(link)
    
    def add_tag(self, tag: str):
        """Add semantic tag"""
        self.tags.add(tag)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "links": self.links,
            "tags": list(self.tags),
            "importance_score": self.importance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create from dictionary"""
        node = cls(
            content=data["content"],
            node_type=data["node_type"],
            metadata=data.get("metadata", {}),
            node_id=data["node_id"]
        )
        node.created_at = datetime.fromisoformat(data["created_at"])
        node.updated_at = datetime.fromisoformat(data["updated_at"])
        node.links = data.get("links", [])
        node.tags = set(data.get("tags", []))
        node.importance_score = data.get("importance_score", 0.0)
        return node


class MemoryUpdateOperation:
    """
    Represents a memory update operation following Mem0 framework
    """
    
    def __init__(self, operation_type: str, node: MemoryNode, reason: str = ""):
        self.operation_type = operation_type  # ADD, UPDATE, DELETE, NOOP
        self.node = node
        self.reason = reason
        self.timestamp = datetime.now()


class VectorStore(ABC):
    """Abstract interface for vector storage backends"""
    
    @abstractmethod
    async def add_embedding(self, node_id: str, content: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    async def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        pass
    
    @abstractmethod
    async def delete_embedding(self, node_id: str) -> None:
        pass


class ChromaDBVectorStore(VectorStore):
    """ChromaDB implementation of vector store"""
    
    def __init__(self, collection_name: str = "quara_memory"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "QuARA Zettelkasten Memory"}
        )
    
    async def add_embedding(self, node_id: str, content: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Add embedding to ChromaDB"""
        self.collection.add(
            ids=[node_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar embeddings"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if results['ids'] and results['distances']:
            return list(zip(results['ids'][0], results['distances'][0]))
        return []
    
    async def delete_embedding(self, node_id: str) -> None:
        """Delete embedding from ChromaDB"""
        self.collection.delete(ids=[node_id])


class MockVectorStore(VectorStore):
    """Mock vector store for testing"""
    
    def __init__(self):
        self.embeddings = {}
        self.metadata = {}
    
    async def add_embedding(self, node_id: str, content: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        self.embeddings[node_id] = (content, embedding)
        self.metadata[node_id] = metadata
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        # Mock similarity search - return random results
        import random
        results = []
        for node_id in list(self.embeddings.keys())[:top_k]:
            results.append((node_id, random.random()))
        return results
    
    async def delete_embedding(self, node_id: str) -> None:
        if node_id in self.embeddings:
            del self.embeddings[node_id]
        if node_id in self.metadata:
            del self.metadata[node_id]


class ZettelkastenMemory:
    """
    Zettelkasten-based long-term memory system for QuARA
    
    Implements:
    - Atomic knowledge nodes with semantic linking
    - Vector-based similarity search
    - Mem0-style update operations (ADD, UPDATE, DELETE, NOOP)
    - Dynamic knowledge graph construction
    """
    
    def __init__(self, use_chromadb: bool = True, embedding_model: str = "all-MiniLM-L6-v2"):
        self.nodes: Dict[str, MemoryNode] = {}
        self.project_memories: Dict[str, List[str]] = {}  # project_id -> node_ids
        
        # Vector store setup
        if use_chromadb and CHROMADB_AVAILABLE:
            self.vector_store = ChromaDBVectorStore()
        else:
            self.vector_store = MockVectorStore()
        
        # Embedding model setup
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None
        
        self.logger = logging.getLogger("ZettelkastenMemory")
        
        # Update history for observability
        self.update_history: List[MemoryUpdateOperation] = []
    
    async def store_knowledge(self, content: str, node_type: str, 
                            project_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Store new knowledge using Mem0-style update logic
        """
        
        # Generate embedding for similarity search
        embedding = await self.generate_embedding(content)
        
        # Search for similar existing knowledge
        similar_nodes = await self.find_similar_nodes(content, threshold=0.8)
        
        # Determine update operation based on similarity
        operation = await self.determine_update_operation(content, node_type, similar_nodes, metadata)
        
        # Execute the operation
        result_node_id = await self.execute_update_operation(operation, project_id)
        
        # Log the operation
        self.update_history.append(operation)
        self.logger.info(f"Memory operation: {operation.operation_type} for node {result_node_id}")
        
        return result_node_id
    
    async def generate_embedding(self, content: str) -> List[float]:
        """Generate embedding for content"""
        if self.embedding_model:
            embedding = self.embedding_model.encode(content)
            return embedding.tolist()
        else:
            # Mock embedding for testing
            import random
            return [random.random() for _ in range(384)]
    
    async def find_similar_nodes(self, content: str, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find similar nodes in memory"""
        embedding = await self.generate_embedding(content)
        similar = await self.vector_store.search_similar(embedding, top_k=10)
        
        # Filter by threshold
        return [(node_id, score) for node_id, score in similar if score < threshold]
    
    async def determine_update_operation(self, content: str, node_type: str, 
                                       similar_nodes: List[Tuple[str, float]], 
                                       metadata: Dict[str, Any] = None) -> MemoryUpdateOperation:
        """
        Determine appropriate update operation based on Mem0 logic
        """
        
        if not similar_nodes:
            # No similar content found - ADD new node
            new_node = MemoryNode(content, node_type, metadata)
            return MemoryUpdateOperation("ADD", new_node, "No similar content found")
        
        # Find most similar node
        most_similar_id, similarity_score = similar_nodes[0]
        
        if most_similar_id in self.nodes:
            existing_node = self.nodes[most_similar_id]
            
            # Check if this is contradictory information
            if await self.is_contradictory(content, existing_node.content):
                if await self.is_more_recent_or_reliable(content, existing_node, metadata):
                    # UPDATE existing node with new information
                    updated_node = MemoryNode(content, node_type, metadata, existing_node.node_id)
                    return MemoryUpdateOperation("UPDATE", updated_node, "More recent/reliable information")
                else:
                    # Keep existing, NOOP
                    return MemoryUpdateOperation("NOOP", existing_node, "Existing information is more reliable")
            
            # Similar but complementary information
            if similarity_score < 0.9:  # Very similar
                # UPDATE existing node with enhanced content
                enhanced_content = await self.merge_content(existing_node.content, content)
                updated_node = MemoryNode(enhanced_content, node_type, metadata, existing_node.node_id)
                return MemoryUpdateOperation("UPDATE", updated_node, "Enhanced with complementary information")
            else:
                # Too similar - NOOP
                return MemoryUpdateOperation("NOOP", existing_node, "Content too similar to existing")
        
        # Default to ADD
        new_node = MemoryNode(content, node_type, metadata)
        return MemoryUpdateOperation("ADD", new_node, "Adding new complementary information")
    
    async def is_contradictory(self, new_content: str, existing_content: str) -> bool:
        """Check if new content contradicts existing content"""
        # Simple heuristic - look for contradictory words
        contradiction_pairs = [
            ("positive", "negative"),
            ("increase", "decrease"),
            ("significant", "not significant"),
            ("supports", "refutes"),
            ("confirmed", "rejected")
        ]
        
        new_lower = new_content.lower()
        existing_lower = existing_content.lower()
        
        for word1, word2 in contradiction_pairs:
            if (word1 in new_lower and word2 in existing_lower) or \
               (word2 in new_lower and word1 in existing_lower):
                return True
        
        return False
    
    async def is_more_recent_or_reliable(self, new_content: str, existing_node: MemoryNode, 
                                       new_metadata: Dict[str, Any] = None) -> bool:
        """Determine if new content is more recent or reliable"""
        new_metadata = new_metadata or {}
        
        # Check timestamp
        new_timestamp = new_metadata.get("timestamp", datetime.now())
        if isinstance(new_timestamp, str):
            new_timestamp = datetime.fromisoformat(new_timestamp)
        
        if new_timestamp > existing_node.updated_at:
            return True
        
        # Check reliability score
        new_reliability = new_metadata.get("reliability_score", 0.5)
        existing_reliability = existing_node.metadata.get("reliability_score", 0.5)
        
        return new_reliability > existing_reliability
    
    async def merge_content(self, existing_content: str, new_content: str) -> str:
        """Merge complementary content"""
        # Simple merging strategy
        return f"{existing_content}\n\nAdditional information: {new_content}"
    
    async def execute_update_operation(self, operation: MemoryUpdateOperation, 
                                     project_id: str = None) -> str:
        """Execute the determined update operation"""
        
        if operation.operation_type == "ADD":
            return await self.add_node(operation.node, project_id)
        
        elif operation.operation_type == "UPDATE":
            return await self.update_node(operation.node, project_id)
        
        elif operation.operation_type == "DELETE":
            await self.delete_node(operation.node.node_id)
            return operation.node.node_id
        
        elif operation.operation_type == "NOOP":
            return operation.node.node_id
        
        else:
            raise ValueError(f"Unknown operation type: {operation.operation_type}")
    
    async def add_node(self, node: MemoryNode, project_id: str = None) -> str:
        """Add new node to memory"""
        self.nodes[node.node_id] = node
        
        # Add to vector store
        embedding = await self.generate_embedding(node.content)
        await self.vector_store.add_embedding(
            node.node_id, 
            node.content, 
            embedding, 
            node.metadata
        )
        
        # Associate with project if provided
        if project_id:
            if project_id not in self.project_memories:
                self.project_memories[project_id] = []
            self.project_memories[project_id].append(node.node_id)
        
        return node.node_id
    
    async def update_node(self, updated_node: MemoryNode, project_id: str = None) -> str:
        """Update existing node"""
        if updated_node.node_id in self.nodes:
            # Preserve original creation time
            original_created_at = self.nodes[updated_node.node_id].created_at
            updated_node.created_at = original_created_at
            updated_node.updated_at = datetime.now()
            
            # Update in memory
            self.nodes[updated_node.node_id] = updated_node
            
            # Update in vector store
            embedding = await self.generate_embedding(updated_node.content)
            await self.vector_store.delete_embedding(updated_node.node_id)  # Remove old
            await self.vector_store.add_embedding(
                updated_node.node_id,
                updated_node.content,
                embedding,
                updated_node.metadata
            )
        
        return updated_node.node_id
    
    async def delete_node(self, node_id: str) -> None:
        """Delete node from memory"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            await self.vector_store.delete_embedding(node_id)
            
            # Remove from project associations
            for project_id, node_ids in self.project_memories.items():
                if node_id in node_ids:
                    node_ids.remove(node_id)
    
    async def create_semantic_link(self, source_node_id: str, target_node_id: str, 
                                 link_type: str = "related") -> None:
        """Create semantic link between nodes"""
        if source_node_id in self.nodes:
            self.nodes[source_node_id].add_link(target_node_id, link_type)
    
    async def query_memory(self, query: str, project_id: str = None, 
                         node_types: List[str] = None, top_k: int = 5) -> List[MemoryNode]:
        """Query memory for relevant knowledge"""
        
        # Generate query embedding
        query_embedding = await self.generate_embedding(query)
        
        # Search for similar nodes
        similar_node_ids = await self.vector_store.search_similar(query_embedding, top_k * 2)
        
        # Filter results
        results = []
        for node_id, score in similar_node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Filter by project if specified
                if project_id and project_id in self.project_memories:
                    if node_id not in self.project_memories[project_id]:
                        continue
                
                # Filter by node type if specified
                if node_types and node.node_type not in node_types:
                    continue
                
                results.append(node)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    async def get_project_memory_summary(self, project_id: str) -> Dict[str, Any]:
        """Get summary of memory for a specific project"""
        if project_id not in self.project_memories:
            return {"project_id": project_id, "node_count": 0, "summary": "No memories found"}
        
        node_ids = self.project_memories[project_id]
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        
        # Analyze node types
        node_type_counts = {}
        for node in nodes:
            node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
        
        # Get recent activity
        recent_updates = sorted(nodes, key=lambda n: n.updated_at, reverse=True)[:5]
        
        return {
            "project_id": project_id,
            "node_count": len(nodes),
            "node_types": node_type_counts,
            "recent_updates": [
                {
                    "node_id": node.node_id,
                    "type": node.node_type, 
                    "content_preview": node.content[:100] + "..." if len(node.content) > 100 else node.content,
                    "updated_at": node.updated_at.isoformat()
                }
                for node in recent_updates
            ],
            "total_links": sum(len(node.links) for node in nodes)
        }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get overall memory system statistics"""
        return {
            "total_nodes": len(self.nodes),
            "total_projects": len(self.project_memories),
            "total_operations": len(self.update_history),
            "operation_breakdown": {
                op_type: sum(1 for op in self.update_history if op.operation_type == op_type)
                for op_type in ["ADD", "UPDATE", "DELETE", "NOOP"]
            },
            "node_type_distribution": {
                node_type: sum(1 for node in self.nodes.values() if node.node_type == node_type)
                for node_type in set(node.node_type for node in self.nodes.values())
            }
        }


__all__ = [
    "MemoryNode",
    "MemoryUpdateOperation", 
    "ZettelkastenMemory",
    "VectorStore",
    "ChromaDBVectorStore",
    "MockVectorStore"
]