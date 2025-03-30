from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, List, Optional

# Protocolo para estrategias de generación de prompt
class PromptStrategy(Protocol):
    def generate_prompt(
        self, 
        filename: str, 
        classes: List[str], 
        functions: List[str], 
        imports: List[str]
    ) -> str:
        """Genera un prompt para resumir código."""
        ...

# Estrategia de prompt por defecto
@dataclass
class DefaultPromptStrategy:
    """Estrategia de prompt predeterminada para generación de resúmenes."""
    
    template: str = """
Analiza el siguiente archivo Python llamado '{filename}' y proporciona un resumen técnico para un desarrollador nuevo.

📁 Nombre del archivo: {filename}
📦 Clases: {classes}
🔧 Funciones: {functions}
📥 Imports: {imports}

Por favor, explica:
1. ¿Cuál es el propósito principal de este archivo?
2. ¿Qué hacen sus clases y funciones más importantes?
3. ¿Cómo se relaciona con el resto del proyecto?

Proporciona un resumen técnico conciso y estructurado.
"""
    
    def generate_prompt(
        self, 
        filename: str, 
        classes: List[str], 
        functions: List[str], 
        imports: List[str]
    ) -> str:
        """
        Genera un prompt usando la plantilla predeterminada.
        
        Args:
            filename: Nombre del archivo analizado
            classes: Lista de nombres de clases
            functions: Lista de nombres de funciones
            imports: Lista de importaciones
        
        Returns:
            Prompt generado
        """
        return self.template.format(
            filename=filename,
            classes=', '.join(classes) if classes else 'Ninguna',
            functions=', '.join(functions) if functions else 'Ninguna',
            imports=', '.join(imports) if imports else 'Ninguno'
        ).strip()

# Protocolo para servicio de resumen
class SummaryService(Protocol):
    def generate_summary(self, file_analysis: 'FileAnalysis', filename: str) -> str:
        """Genera un resumen para un archivo."""
        ...

# Clase de servicio de resumen principal
class CodeSummarizer:
    """
    Servicio de generación de resúmenes de código con alto grado de flexibilidad.
    
    Características:
    - Inyección de dependencias para LLM
    - Estrategia de generación de prompt configurable
    - Separación clara de responsabilidades
    """
    
    def __init__(
        self, 
        llm: 'BaseLLM',
        prompt_strategy: Optional[PromptStrategy] = None
    ):
        """
        Inicializa el servicio de resumen.
        
        Args:
            llm: Modelo de lenguaje para generar resúmenes
            prompt_strategy: Estrategia personalizada para generación de prompts
        """
        self._llm = llm
        self._prompt_strategy = prompt_strategy or DefaultPromptStrategy()
    
    def generate_summary(
        self, 
        file_analysis: 'FileAnalysis', 
        filename: str
    ) -> str:
        """
        Genera un resumen para un archivo de código.
        
        Args:
            file_analysis: Análisis del archivo
            filename: Nombre del archivo
        
        Returns:
            Resumen generado por el modelo
        """
        # Extracción de información
        classes = [cls.name for cls in file_analysis.classes]
        functions = [fn.name for fn in file_analysis.functions]
        imports = file_analysis.imports
        
        # Generación de prompt usando estrategia
        prompt = self._prompt_strategy.generate_prompt(
            filename, classes, functions, imports
        )
        
        # Generación de resumen
        return self._llm.ask_with_retry(prompt)
    
    def set_prompt_strategy(self, strategy: PromptStrategy):
        """
        Permite cambiar dinámicamente la estrategia de generación de prompt.
        
        Args:
            strategy: Nueva estrategia de generación de prompt
        """
        self._prompt_strategy = strategy

# Ejemplo de estrategia de prompt personalizada
class DetailedPromptStrategy:
    """Estrategia de prompt más detallada."""
    
    def generate_prompt(
        self, 
        filename: str, 
        classes: List[str], 
        functions: List[str], 
        imports: List[str]
    ) -> str:
        return f"""
Realizar un análisis profundo del archivo Python: {filename}

🔍 Análisis Detallado:
- Nombre del archivo: {filename}
- Clases detectadas: {', '.join(classes) or 'No se encontraron clases'}
- Funciones identificadas: {', '.join(functions) or 'No se encontraron funciones'}
- Importaciones: {', '.join(imports) or 'Sin importaciones'}

Proporciona un análisis técnico exhaustivo que incluya:
1. Arquitectura y diseño del módulo
2. Relaciones entre componentes
3. Posibles patrones de diseño
4. Recomendaciones de mejora
5. Contexto dentro de la arquitectura del proyecto

Mantén un tono técnico y objetivo.
""".strip()