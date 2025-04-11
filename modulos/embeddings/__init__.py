"""
Módulo para la generación y manejo de embeddings.
Proporciona componentes para trabajar con diferentes modelos de embeddings.
"""

from modulos.embeddings.embeddings_factory import EmbeddingFactory

# Exportar la función para reiniciar instancias
reset_embedding_managers = EmbeddingFactory.reset_instances
