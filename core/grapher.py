'''# core/grapher.py
import networkx as nx
from pyvis.network import Network
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("grapher")

class CodeGraphGenerator:
    """
    Genera un grafo visual que representa relaciones entre archivos, clases, 
    funciones e importaciones en un proyecto de código.
    """
    
    def __init__(self, repo_results: Dict[str, Any]):
        """
        Inicializa el generador de grafos.
        
        Args:
            repo_results: Diccionario con resultados del análisis de código por archivo
        """
        self.repo_results = repo_results
        self.graph = nx.DiGraph()
        
    def build_graph(self) -> nx.DiGraph:
        """
        Construye el grafo basado en los resultados del análisis.
        
        Returns:
            El grafo dirigido construido
        """
        try:
            for file, analysis in self.repo_results.items():
                # Añadir nodo de archivo
                self.graph.add_node(file, type="file", title=f"File: {file}")
                
                # Añadir nodos de clases y funciones
                for cls in analysis.classes:
                    self.graph.add_node(cls.name, type="class", title=f"Class: {cls.name}")
                    self.graph.add_edge(file, cls.name)
                
                for func in analysis.functions:
                    self.graph.add_node(func.name, type="function", title=f"Function: {func.name}")
                    self.graph.add_edge(file, func.name)
                
                # Añadir dependencias de imports
                for imp in analysis.imports:
                    self.graph.add_node(imp, type="import", title=f"Import: {imp}")
                    self.graph.add_edge(file, imp)
                    
            logger.info(f"Grafo construido con {len(self.graph.nodes)} nodos y {len(self.graph.edges)} aristas")
            return self.graph
            
        except Exception as e:
            logger.error(f"Error al construir el grafo: {str(e)}")
            raise
            
    def generate_html(self, output_path: Optional[str] = None) -> str:
        """
        Genera un archivo HTML interactivo del grafo.
        
        Args:
            output_path: Ruta opcional donde guardar el archivo HTML
            
        Returns:
            Ruta del archivo HTML generado
        """
        self.build_graph()
        
        # Crear red visual
        net = Network(height="700px", width="100%", directed=True, notebook=False)
        
        # Personalizar nodos por tipo
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            
            if node_type == 'file':
                net.add_node(node, label=Path(node).name, color='#6699CC', size=15)
            elif node_type == 'class':
                net.add_node(node, color='#99CC66', size=12)
            elif node_type == 'function':
                net.add_node(node, color='#CC9966', size=10)
            elif node_type == 'import':
                net.add_node(node, color='#CC6699', size=8)
            else:
                net.add_node(node, color='#CCCCCC')
                
        # Añadir aristas
        for source, target in self.graph.edges():
            net.add_edge(source, target)
            
        # Configurar opciones visuales
        net.set_options("""
        var options = {
            "nodes": {
                "font": {
                    "size": 12
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                },
                "color": {
                    "inherit": "from"
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            },
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 100,
                    "springConstant": 0.01,
                    "nodeDistance": 120
                },
                "solver": "hierarchicalRepulsion"
            },
            "interaction": {
                "navigationButtons": true,
                "keyboard": true
            }
        }
        """)
        
        # Generar archivo HTML
        try:
            if output_path:
                # Usar ruta proporcionada
                file_path = output_path
                net.save_graph(file_path)
                logger.info(f"Grafo guardado en {file_path}")
            else:
                # Usar archivo temporal
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    file_path = tmp.name
                    net.save_graph(file_path)
                    logger.info(f"Grafo temporal guardado en {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error al generar HTML: {str(e)}")
            raise  '''
            
# core/grapher.py
import networkx as nx
from pyvis.network import Network
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("grapher")

class CodeGraphGenerator:
    """
    Genera un grafo visual que representa relaciones entre archivos, clases, 
    funciones e importaciones en un proyecto de código.
    """
    
    def __init__(self, repo_results: Dict[str, Any]):
        self.repo_results = repo_results
        self.graph = nx.DiGraph()
        
    def build_graph(self) -> nx.DiGraph:
        """
        Construye el grafo basado en los resultados del análisis.
        
        Returns:
            El grafo dirigido construido
        """
        try:
            for file, analysis in self.repo_results.items():
                # Añadir nodo de archivo
                file_name = Path(file).name
                self.graph.add_node(
                    file, 
                    type="file", 
                    label=f"📄 {file_name}",
                    title=f"File: {file}\nPath: {file}",
                    level=0
                )
                
                # Añadir nodos de clases
                for cls in analysis.classes:
                    self.graph.add_node(
                        cls.name, 
                        type="class", 
                        label=f"🟢 {cls.name}",
                        title=f"Class: {cls.name}\nFile: {file}",
                        level=1
                    )
                    self.graph.add_edge(file, cls.name, label="contains")
                
                # Añadir nodos de funciones
                for func in analysis.functions:
                    func_name = func.name.split('.')[-1]  # Mostrar solo el nombre final
                    # Tooltip seguro que no requiere return_type
                    func_tooltip = f"Function: {func.name}\nFile: {file}"
                    
                    # Intentar obtener return_type si existe, sin fallar si no
                    if hasattr(func, 'return_type'):
                        func_tooltip += f"\nReturns: {func.return_type}"
                    
                    self.graph.add_node(
                        func.name, 
                        type="function", 
                        label=f"🔵 {func_name}",
                        title=func_tooltip,
                        level=2
                    )
                    self.graph.add_edge(file, func.name, label="contains")
                
                # Añadir dependencias de imports
                for imp in analysis.imports:
                    imp_name = imp.split('.')[-1]  # Mostrar solo el último componente
                    self.graph.add_node(
                        imp, 
                        type="import", 
                        label=f"🟣 {imp_name}",
                        title=f"Import: {imp}\nUsed in: {file}",
                        level=3
                    )
                    self.graph.add_edge(file, imp, label="uses", dashes=True)
                    
            logger.info(f"Grafo construido con {len(self.graph.nodes)} nodos y {len(self.graph.edges)} aristas")
            return self.graph
            
        except Exception as e:
            logger.error(f"Error al construir el grafo: {str(e)}")
            raise
            
    def generate_html(self, output_path: Optional[str] = None, show_imports: bool = True) -> str:
        """
        Genera un archivo HTML interactivo del grafo.
        
        Args:
            output_path: Ruta opcional donde guardar el archivo HTML
            show_imports: Si se deben mostrar los nodos de importación
            
        Returns:
            Ruta del archivo HTML generado
        """
        self.build_graph()
        
        # Crear red visual
        net = Network(
            height="800px", 
            width="100%", 
            directed=True, 
            notebook=False,
            bgcolor="#f5f5f5",
            font_color="#333333"
        )
        
        # Personalizar nodos por tipo con mejor diseño
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            
            if node_type == 'file':
                net.add_node(
                    node,
                    label=attrs.get('label', node),
                    color='#4285F4',
                    size=25,
                    shape='box',
                    font={'size': 14, 'face': 'arial', 'bold': True},
                    level=attrs.get('level', 0)
                )
            elif node_type == 'class':
                net.add_node(
                    node,
                    label=attrs.get('label', node),
                    color='#0F9D58',
                    size=20,
                    shape='ellipse',
                    font={'size': 12, 'face': 'arial'},
                    level=attrs.get('level', 1)
                )
            elif node_type == 'function':
                net.add_node(
                    node,
                    label=attrs.get('label', node),
                    color='#DB4437',
                    size=15,
                    shape='diamond',
                    font={'size': 11, 'face': 'arial'},
                    level=attrs.get('level', 2)
                )
            elif node_type == 'import' and show_imports:
                net.add_node(
                    node,
                    label=attrs.get('label', node),
                    color='#9E9E9E',
                    size=10,
                    shape='triangle',
                    font={'size': 10, 'face': 'arial'},
                    level=attrs.get('level', 3)
                )
                
        # Añadir aristas con estilos diferentes
        for source, target, attrs in self.graph.edges(data=True):
            edge_label = attrs.get('label', '')
            if 'dashes' in attrs and attrs['dashes']:
                net.add_edge(source, target, label=edge_label, dashes=[5,5], width=1, color='#9E9E9E')
            else:
                net.add_edge(source, target, label=edge_label, width=2)
            
        # Configurar opciones visuales mejoradas
        net.set_options("""
        {
            "nodes": {
                "borderWidth": 1,
                "borderWidthSelected": 2,
                "shadow": {
                    "enabled": true,
                    "color": "rgba(0,0,0,0.2)",
                    "size": 10,
                    "x": 5,
                    "y": 5
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.8
                    }
                },
                "color": {
                    "inherit": false,
                    "opacity": 0.8
                },
                "smooth": {
                    "type": "cubicBezier",
                    "forceDirection": "horizontal",
                    "roundness": 0.4
                },
                "font": {
                    "size": 10,
                    "align": "middle"
                }
            },
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 150,
                    "springConstant": 0.01,
                    "nodeDistance": 160,
                    "damping": 0.09
                },
                "solver": "hierarchicalRepulsion",
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000,
                    "updateInterval": 25
                }
            },
            "interaction": {
                "navigationButtons": true,
                "keyboard": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": true,
                "multiselect": true
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "nodeSpacing": 150,
                    "levelSeparation": 120
                }
            }
        }
        """)
        
        # Generar archivo HTML
        try:
            if output_path:
                file_path = output_path
                net.save_graph(file_path)
                logger.info(f"Grafo guardado en {file_path}")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    file_path = tmp.name
                    net.save_graph(file_path)
                    logger.info(f"Grafo temporal guardado en {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error al generar HTML: {str(e)}")
            raise